"""Training data: study -> bag of 2.5D windows read from the flat slice cache; augmentation runs on the GPU.

The Dataset only picks slices (uint8, cached 384 px / 140 mm) and describes each window: [view slot, slice offset
from the series centre in mm]. `gpu_views` then builds, per window, the affine that crops / zooms / rotates / shifts
the cached slice to the model resolution (one affine per study x view slot, shared by its windows), samples it with
grid_sample and applies the intensity augmentation. The same affine is passed to the model so tokens keep their
anatomical positions. A zoomed view (ROI) is an extra view slot: windows of its parent series, cropped on the GPU.

Window samplers (training bags; evaluation always uses every centre of the series span):
- FlatBags ("fixed"): k windows per series (k by series type), stratified-jittered over the series span, plus k per
  zoomed view.
- BudgetBags ("progressive" / "budget"): a per-study window budget K split over the present series by weight
  (largest-remainder allocation capped by each series' number of centres), K changing with the epoch, and optional
  instance dropout of the built windows.
"""
import math, random
import multiprocessing as mp
import numpy as np, pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SLOTS = ["SAG_FS", "SAG_N", "COR_FS", "COR_N", "AX_FS", "AX_N"]
PLANE = {"SAG": 0, "COR": 1, "AX": 2}


def parse_rois(spec):
    """'SLOT:y0:y1:x0:x1:k;...' (fractions of the cached 140 mm slice, k training windows) -> list of dicts.
    ROI i is an extra view with slot id 6 + i: the same series, cropped and enlarged on the GPU (gpu_views)."""
    out = []
    for item in [x for x in spec.split(";") if x]:
        f = item.split(":")
        out.append(dict(parent=f[0], y0=float(f[1]), y1=float(f[2]), x0=float(f[3]), x1=float(f[4]),
                        k=int(f[5]) if len(f) > 5 else 3, plane=PLANE[f[0].split("_")[0]]))
    return out


def roi_tensor(rois, device):
    """(R, 4) [centre x, centre y, half-width, half-height] in normalised [-1, 1] coordinates."""
    if not rois:
        return None
    return torch.tensor([[r["x0"] + r["x1"] - 1, r["y0"] + r["y1"] - 1, r["x1"] - r["x0"], r["y1"] - r["y0"]]
                         for r in rois], dtype=torch.float32, device=device)


class FlatBags(Dataset):
    def __init__(self, uids, labels, index_csv, flat_npy, k_train, train, eval_stride=2, span=(0.04, 0.96),
                 slot_drop=0.0, thick_p=0.0, slots=SLOTS, rois=(), win_step=1):
        """k_train: dict slot -> windows per series in training (stratified jitter over the span).
        Evaluation: every eval_stride-th centre of the span. slots: series used (ROI views of an unused parent are
        dropped too). win_step: stored-slice spacing of the 3 slices of a window (thick_p doubles it at random in
        training: thicker-slice simulation)."""
        self.uids, self.labels = list(uids), labels
        self.flat_npy, self.mm = flat_npy, None
        ix = pd.read_csv(index_csv, dtype={"uid": str})
        ix = ix[ix.slot.isin(slots) & ix.uid.isin(set(self.uids))]
        self.rows = {u: g for u, g in ix.groupby("uid")}
        self.k_train, self.train, self.eval_stride, self.span = k_train, train, eval_stride, span
        self.slot_drop, self.thick_p = slot_drop, thick_p
        self.slot_id = {s: i for i, s in enumerate(SLOTS)}
        self.rois = list(rois)
        self.win_step = int(win_step)

    def __len__(self):
        return len(self.uids)

    def _centres(self, n, k):
        lo, hi = max(1, int(round(self.span[0] * (n - 1)))), min(n - 2, int(round(self.span[1] * (n - 1))))
        cs = np.arange(lo, hi + 1) if hi >= lo else np.array([n // 2])
        if self.train:
            edges = np.linspace(0, len(cs), k + 1)
            return np.array([cs[min(random.randrange(int(edges[j]), max(int(edges[j]) + 1, int(edges[j + 1]))), len(cs) - 1)]
                             for j in range(k)])
        return cs[::self.eval_stride] if self.eval_stride > 1 else cs

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = np.load(self.flat_npy, mmap_mode="r")
        uid = self.uids[i]
        g = self.rows.get(uid)
        recs = [] if g is None else list(g.itertuples(index=False))
        if self.train and self.slot_drop > 0 and len(recs) > 2:   # series dropout, keeping >= 2 and a fat-suppressed one
            for _ in range(10):
                keep = [r for r in recs if random.random() >= self.slot_drop]
                if len(keep) >= 2 and any(r.slot.endswith("FS") for r in keep):
                    recs = keep; break
        wins, meta = [], []
        for r in recs:
            n = int(r.n)
            views = [(self.slot_id[r.slot], self.k_train.get(r.slot, 0))] if (not self.train or r.slot in self.k_train) else []
            views += [(len(SLOTS) + i, ro["k"]) for i, ro in enumerate(self.rois) if ro["parent"] == r.slot]
            if not views:
                continue
            step = self.win_step * (2 if (self.train and self.thick_p > 0 and random.random() < self.thick_p) else 1)
            step_mm = (r.thick if r.thick > 0 else 3.5) * max(r.n_orig, n) / n
            block, lo = None, 0
            for sid, k in views:
                cs = self._centres(n, k)
                idx = np.clip(cs[:, None] + step * np.arange(-1, 2)[None], 0, n - 1)
                if block is None:
                    lo = 0 if len(views) > 1 else int(idx.min())
                    hi = n - 1 if len(views) > 1 else int(idx.max())
                    block = np.asarray(self.mm[int(r.off) + lo:int(r.off) + hi + 1])   # one contiguous read
                wins.append(block[idx - lo])                                           # (k, 3, H, W) uint8
                meta.append(np.stack([np.full(len(cs), sid, np.float32),
                                      ((cs - (n - 1) / 2) * step_mm).astype(np.float32)], 1))
        if not wins:
            wins, meta = [np.zeros((1, 3, 384, 384), np.uint8)], [np.zeros((1, 2), np.float32)]
        y = np.array(self.labels[uid], np.float32)          # copy: pandas .values are read-only
        return torch.from_numpy(np.concatenate(wins)), torch.from_numpy(np.concatenate(meta)), torch.from_numpy(y), i


def hamilton(total, w, cap):
    """Largest-remainder allocation of `total` over weights w with capacities cap (saturated budget redistributed)."""
    w, cap = np.asarray(w, float), np.asarray(cap, int); out = np.zeros(len(w), int)
    left = min(int(total), int(cap[w > 0].sum()))
    while left:
        act = (cap > out) & (w > 0)
        if not act.any():
            break
        ideal = np.where(act, w, 0); ideal = ideal / ideal.sum() * left
        fl = np.minimum(np.floor(ideal).astype(int), cap - out); out += fl; left -= int(fl.sum())
        if not left:
            break
        for i in sorted([i for i in np.flatnonzero(act) if cap[i] > out[i]], key=lambda i: (-(ideal[i] - np.floor(ideal[i])), i))[:left]:
            out[i] += 1; left -= 1
    return out


class BudgetBags(FlatBags):
    """Training bags with a per-study window budget (evaluation bags are those of FlatBags).
    weights: dict series -> allocation weight; schedule: [(K, epochs), ...] with K an int or "ALL" (every centre of
    every series), the last entry kept after the schedule ends; drop: instance dropout of the built windows (keep >= 1).
    The epoch is read from a sample counter shared by the forked loader workers (samples served // training-set size),
    so the K phase of a few prefetched samples at an epoch boundary depends on worker timing."""
    def __init__(self, *a, weights=None, schedule=(("ALL", 1),), drop=0.0, **k):
        super().__init__(*a, **k)
        self.weights, self.schedule, self.drop = dict(weights), list(schedule), float(drop)
        self.count = mp.Value("l", 0)            # shared across forked loader workers -> samples served so far

    def _phase(self, c):
        ep = c // max(1, len(self.uids))
        for kk, n in self.schedule:
            if ep < n:
                return kk
            ep -= n
        return self.schedule[-1][0]

    def _budget(self):
        with self.count.get_lock():
            c = self.count.value; self.count.value += 1
        return self._phase(c)

    def __getitem__(self, i):
        if not self.train:
            return super().__getitem__(i)
        if self.mm is None:
            self.mm = np.load(self.flat_npy, mmap_mode="r")
        uid = self.uids[i]; g = self.rows.get(uid)
        recs = [] if g is None else list(g.itertuples(index=False))
        if self.slot_drop > 0 and len(recs) > 2:
            for _ in range(10):
                keep = [r for r in recs if random.random() >= self.slot_drop]
                if len(keep) >= 2 and any(r.slot.endswith("FS") for r in keep):
                    recs = keep; break
        ncen = {r.slot: len(self._centres_all(int(r.n))) for r in recs}
        budget = self._budget()
        if budget == "ALL":
            alloc = dict(ncen)
        else:
            present = [s for s in SLOTS if s in ncen]
            # budget of the present series = their share of K under all 6 weights, then capped redistribution
            full = hamilton(int(budget), [self.weights[s] for s in SLOTS], [10 ** 6] * 6)
            req = int(sum(full[SLOTS.index(s)] for s in present))
            alloc = dict(zip(present, hamilton(req, [self.weights[s] for s in present], [ncen[s] for s in present])))
        wins, meta = [], []
        for r in recs:
            n = int(r.n)
            views = [(self.slot_id[r.slot], int(alloc.get(r.slot, 0)))]
            views += [(len(SLOTS) + j, ro["k"]) for j, ro in enumerate(self.rois) if ro["parent"] == r.slot]
            step = self.win_step * (2 if (self.thick_p > 0 and random.random() < self.thick_p) else 1)
            step_mm = (r.thick if r.thick > 0 else 3.5) * max(r.n_orig, n) / n
            block = np.asarray(self.mm[int(r.off):int(r.off) + n])
            for sid, k in views:
                if k <= 0:
                    continue
                cs = self._centres(n, k)
                idx = np.clip(cs[:, None] + step * np.arange(-1, 2)[None], 0, n - 1)
                wins.append(block[idx])
                meta.append(np.stack([np.full(len(cs), sid, np.float32), ((cs - (n - 1) / 2) * step_mm).astype(np.float32)], 1))
        if not wins:
            wins, meta = [np.zeros((1, 3, 384, 384), np.uint8)], [np.zeros((1, 2), np.float32)]
        x, m = np.concatenate(wins), np.concatenate(meta)
        if self.drop > 0:
            keep = np.random.random(len(x)) >= self.drop
            if not keep.any():
                keep[np.random.randint(len(keep))] = True
            x, m = x[keep], m[keep]
        y = np.array(self.labels[uid], np.float32)
        return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y), i

    def _centres_all(self, n):
        lo, hi = max(1, int(round(self.span[0] * (n - 1)))), min(n - 2, int(round(self.span[1] * (n - 1))))
        return np.arange(lo, hi + 1) if hi >= lo else np.array([n // 2])

    def _centres(self, n, k):
        if not self.train:
            return super()._centres(n, k)
        cs = self._centres_all(n)
        if k >= len(cs):                          # ALL / saturated: every unique centre once
            return cs
        edges = np.linspace(0, len(cs), k + 1)
        return np.array([cs[min(random.randrange(int(edges[j]), max(int(edges[j]) + 1, int(edges[j + 1]))), len(cs) - 1)] for j in range(k)])


def collate(batch):
    n = max(b[0].shape[0] for b in batch)
    B, (C, H, W) = len(batch), batch[0][0].shape[1:]
    X = torch.zeros(B, n, C, H, W, dtype=torch.uint8)
    Mt = torch.zeros(B, n, 2)
    M = torch.zeros(B, n, dtype=torch.bool)
    for j, (x, m, _, _) in enumerate(batch):
        X[j, :len(x)] = x; Mt[j, :len(x)] = m; M[j, :len(x)] = True
    return X, Mt, M, torch.stack([b[2] for b in batch]), torch.tensor([b[3] for b in batch])


AUG = {   # rotation deg, zoom range (scale of the field of view), shift (fraction), p(affine), intensity level
    "none": (0, (1.0, 1.0), 0.0, 0.0, 0),
    "strong": (15, (0.85, 1.15), 0.08, 0.9, 2),
}


def gpu_views(X, Mt, M, res, aug="none", rois=None):
    """X (B, W, 3, 384, 384) uint8 on GPU -> (B, W, 3, res, res) float in [0, 1] and theta (B, W, 2, 3).
    rois: roi_tensor() for the zoomed views (slot ids 6 + i), which show their crop of the parent slice enlarged to
    res. One random affine per (study, view slot)."""
    B, W = M.shape
    dev = X.device
    rot, (z0, z1), sh, p_aff, lvl = AUG[aug]
    R = 0 if rois is None else len(rois)
    S = 6 + R
    ang = torch.zeros(B, S, device=dev); sc = torch.ones(B, S, device=dev); tr = torch.zeros(B, S, 2, device=dev)
    if p_aff > 0:
        on = (torch.rand(B, S, device=dev) < p_aff).float()
        ang = on * (torch.rand(B, S, device=dev) * 2 - 1) * math.radians(rot)
        sc = 1 + on * (torch.exp(torch.empty(B, S, device=dev).uniform_(math.log(z0), math.log(z1))) - 1)
        tr = on[..., None] * (torch.rand(B, S, 2, device=dev) * 2 - 1) * sh * 2
    s = 1.0 / sc                                                         # >1 zoom-in shows less of the field
    if R:
        s = torch.cat([s[:, :6], 1 / sc[:, 6:]], 1)                      # zoomed views: the crop sets the field
    cos, sin = torch.cos(ang) * s, torch.sin(ang) * s
    th = torch.stack([torch.stack([cos, -sin, tr[..., 0]], -1), torch.stack([sin, cos, tr[..., 1]], -1)], -2)  # (B, S, 2, 3)
    if R:   # source = crop centre + half-size * (augmentation affine of the output position)
        th = th.clone()
        th[:, 6:, 0, :] = th[:, 6:, 0, :] * rois[None, :, 2:3]
        th[:, 6:, 1, :] = th[:, 6:, 1, :] * rois[None, :, 3:4]
        th[:, 6:, :, 2] = th[:, 6:, :, 2] + rois[None, :, :2]
    slot = Mt[..., 0].long().clamp(0, S - 1)
    theta = torch.gather(th, 1, slot[..., None, None].expand(B, W, 2, 3))   # (B, W, 2, 3)
    xf = X[M].float() / 255.0                                            # (n, 3, H, W)
    tf = theta[M]
    grid = F.affine_grid(tf, (len(xf), 3, res, res), align_corners=False)
    out = F.grid_sample(xf, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if lvl:
        out = _intensity(out, lvl)
    full = torch.zeros(B, W, 3, res, res, device=dev, dtype=out.dtype)
    full[M] = out
    return full, theta


def _intensity(x, lvl):
    """Per-window MRI intensity augmentation (lvl 2 = "strong"): gain, contrast, gamma, low-resolution simulation,
    Rician noise."""
    n = x.shape[0]
    dev = x.device
    r = lambda: torch.rand(n, 1, 1, 1, device=dev)
    g = 1 + (r() * 2 - 1) * (0.10 if lvl == 1 else 0.15)                 # gain
    x = x * g
    if lvl >= 2:
        m = x.mean((1, 2, 3), keepdim=True)
        x = (x - m) * (1 + (r() * 2 - 1) * 0.15) + m                     # contrast
        gam = torch.exp((r() * 2 - 1) * 0.3)
        x = torch.where(r() < 0.5, x.clamp(0, 1) ** gam, x)              # gamma
        lowres = r() < 0.15                                              # low-resolution simulation
        if lowres.any():
            idx = lowres.flatten().nonzero().flatten()
            H = x.shape[-1]
            f = random.uniform(0.5, 0.75)
            sm = F.interpolate(F.interpolate(x[idx], scale_factor=f, mode="bilinear", align_corners=False),
                               size=(H, H), mode="bilinear", align_corners=False)
            x = x.index_copy(0, idx, sm)
        sg = (r() < 0.2).float() * r() * 0.03                            # Rician noise
        x = torch.sqrt((x.clamp(0, 1) + sg * torch.randn_like(x)) ** 2 + (sg * torch.randn_like(x)) ** 2)
    return x.clamp(0, 1)
