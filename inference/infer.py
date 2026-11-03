"""Inference: DICOM -> 2.5D windows -> four (or eight) models -> per-label rank mean -> submission.csv.

- Preprocessing (preprocess.py) runs in a pool of CPU processes, a few studies ahead of the GPUs.
- One worker thread per GPU, each with its own copy of every model; fp16 autocast.
- Windows: every slice of every series is a window centre (dense stride 1); each zoomed view uses the windows of
  its parent series and crops them on the GPU.
- Ensemble: per label, the mean over the models of their percentile ranks across the test studies.
- Paths: environment variables COMP_DIR / MODEL_DIR / OUT, else SETTINGS.json at the repository root, else the
  Kaggle input folders (the notebook): the weights from the attached Kaggle Model songqizhou/rsna-knee-7th-place,
  else from any attached input that has them (the dataset songqizhou/rsna-knee-7th-place-models, the same files).
usage: [ENSEMBLE=4|8] python inference/infer.py
"""
import os, sys, glob, json, time, queue, threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True   # keep the output folder clean
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_study  # noqa: E402
from model import KQNet  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SERIES = ["SAG_FS", "SAG_N", "COR_FS", "COR_N", "AX_FS", "AX_N"]
PLANE = {"SAG": 0, "COR": 1, "AX": 2}
# 13 zoomed views: (parent series, y0, y1, x0, x1) as fractions of the 140 mm slice. They cover the menisci, the
# patellofemoral joint and the medial / lateral compartments, each enlarged to 384 px.
ROIS = [("AX_FS", 0.0, 0.6, 0.2, 0.8), ("SAG_FS", 0.1, 0.7, 0.0, 0.6), ("COR_FS", 0.2, 0.8, 0.4, 1.0),
        ("COR_FS", 0.2, 0.8, 0.0, 0.6), ("SAG_FS", 0.38, 0.83, 0.1, 0.55), ("SAG_FS", 0.38, 0.83, 0.45, 0.9),
        ("SAG_N", 0.38, 0.83, 0.45, 0.9), ("COR_FS", 0.3, 0.75, 0.08, 0.53), ("COR_FS", 0.3, 0.75, 0.47, 0.92),
        ("AX_FS", 0.02, 0.5, 0.2, 0.8), ("AX_N", 0.02, 0.5, 0.2, 0.8), ("SAG_FS", 0.12, 0.62, 0.08, 0.58),
        ("SAG_N", 0.12, 0.62, 0.08, 0.58)]
# The two selected submissions, models in the order their percentile ranks are averaged. Final submission 1: the four
# models. Final submission 2: the same four + a twin of each (another seed, trained on the training set + the 58 gold
# studies). Both scored 0.970 public / 0.957 private. ENSEMBLE=4 (default) or ENSEMBLE=8 picks one.
ENSEMBLES = {"4": ["effnetv2s_fixed.pt", "effnetv2s_progressive.pt", "convnext_small.pt", "convnext_tiny.pt"],
             "8": ["effnetv2s_fixed.pt", "effnetv2s_fixed_twin.pt", "effnetv2s_progressive.pt",
                   "effnetv2s_progressive_twin.pt", "convnext_small.pt", "convnext_small_twin.pt", "convnext_tiny.pt",
                   "convnext_tiny_twin.pt"]}
MODELS = ENSEMBLES[os.environ.get("ENSEMBLE", "4")]
SPAN = (0.04, 0.96)   # window centres cover the central 92% of each series
RES = 384
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
TIME_BUDGET_S = 8 * 3600   # past this, stop and submit the studies done so far (the limit is 9 h)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setting(key):
    """A path from SETTINGS.json at the repository root (relative paths are relative to that root), or None."""
    f = os.path.join(ROOT, "SETTINGS.json")
    if not os.path.exists(f):
        return None
    v = json.load(open(f)).get(key)
    return v if v is None or os.path.isabs(v) else os.path.join(ROOT, v)


def find_dir(fname):
    for depth in range(1, 6):
        hits = glob.glob("/kaggle/input/" + "*/" * depth + fname)
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError(fname)


KAGGLE_MODEL = "rsna-knee-7th-place"   # Kaggle Model songqizhou/rsna-knee-7th-place (PyTorch, variation default)


def find_model_dir(fname):
    """The folder of the attached Kaggle Model (its newest version), else any attached input with the weights."""
    roots = [r for depth in range(3) for r in glob.glob("/kaggle/input/" + "*/" * depth + KAGGLE_MODEL)]
    hits = [h for r in roots for h in glob.glob(f"{r}/**/{fname}", recursive=True)]
    if hits:
        return os.path.dirname(max(hits, key=lambda h: [int(x) if x.isdigit() else -1 for x in h.split("/")]))
    return find_dir(fname)


def study_windows(vols):
    """{slot: (volume (n, 384, 384) uint8, slices in the series, thickness mm)} -> windows (W, 3, 384, 384) uint8
    and meta (W, 2) float32 = [view slot, slice offset from the series centre in mm]."""
    wins, meta = [], []
    for slot in SERIES:
        if slot not in vols:
            continue
        v, n_orig, thick = vols[slot]
        if v.ndim != 3 or v.shape[1:] != (RES, RES) or len(v) < 1:
            continue
        n = len(v)
        lo, hi = max(1, int(round(SPAN[0] * (n - 1)))), min(n - 2, int(round(SPAN[1] * (n - 1))))
        cs = np.arange(lo, hi + 1) if hi >= lo else np.array([n // 2])
        idx = np.clip(cs[:, None] + np.arange(-1, 2)[None], 0, n - 1)    # 3 neighbouring slices per window
        step_mm = (thick if thick > 0 else 3.5) * max(n_orig, n) / n       # spacing of the kept slices
        off = ((cs - (n - 1) / 2) * step_mm).astype(np.float32)
        for sid in [SERIES.index(slot)] + [6 + i for i, r in enumerate(ROIS) if r[0] == slot]:
            wins.append(v[idx])
            meta.append(np.stack([np.full(len(cs), sid, np.float32), off], 1))
    if not wins:
        return None
    return np.concatenate(wins), np.concatenate(meta)


def view_affines(device):
    """(6 + 13, 2, 3) affine per view slot, in normalised [-1, 1] coordinates of the cached slice: a series shows the
    whole slice; zoomed view i shows its crop, enlarged to the model resolution."""
    th = torch.zeros(6 + len(ROIS), 2, 3)
    th[:6, 0, 0] = th[:6, 1, 1] = 1.0
    crop = torch.tensor([[x0 + x1 - 1, y0 + y1 - 1, x1 - x0, y1 - y0] for _, y0, y1, x0, x1 in ROIS], dtype=torch.float32)
    th[6:, 0, 0], th[6:, 1, 1] = crop[:, 2], crop[:, 3]                    # half-size of the crop
    th[6:, 0, 2], th[6:, 1, 2] = crop[:, 0], crop[:, 1]                    # centre of the crop
    return th.to(device)


def gpu_views(X, Mt, M, affines):
    """Windows X (B, W, 3, 384, 384) uint8 on the GPU -> model input (B, W, 3, RES, RES) in [0, 1] and theta
    (B, W, 2, 3), the affine of each window's view, which the model also uses to place its tokens."""
    B, W = M.shape
    theta = affines[Mt[..., 0].long().clamp(0, len(affines) - 1)]          # (B, W, 2, 3)
    xf = X[M].float() / 255.0
    grid = F.affine_grid(theta[M], (len(xf), 3, RES, RES), align_corners=False)
    out = F.grid_sample(xf, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    full = torch.zeros(B, W, 3, RES, RES, device=X.device, dtype=out.dtype)
    full[M] = out
    return full, theta


def load_models(model_dir, devs):
    """-> one list per model with a copy on every device."""
    roi_planes = [PLANE[r[0].split("_")[0]] for r in ROIS]
    models = []
    for f in MODELS:
        ck = torch.load(os.path.join(model_dir, f), map_location="cpu", weights_only=True)
        cfg = ck["config"]
        copies = []
        for dev in devs:
            m = KQNet(cfg["arch"], roi_planes, D=cfg["dim"])
            m.load_state_dict(ck["model"])
            m.set_att_temp(cfg["att_temp"])
            m.eval().to(dev)
            if dev != "cpu":
                m.bb.to(memory_format=torch.channels_last)
            copies.append(m)
        models.append(copies)
        print(f"{f}: {cfg['arch']}, attention temperature {cfg['att_temp']}", flush=True)
    return models


@torch.no_grad()
def predict_study(models, di, affines, wins, meta):
    """-> (n_models, 12) probabilities for one study, on device index di."""
    dev = affines.device
    X = torch.from_numpy(wins)[None].to(dev, non_blocking=True)
    Mt = torch.from_numpy(meta)[None].to(dev)
    M = torch.ones(1, len(wins), dtype=torch.bool, device=dev)
    xv, theta = gpu_views(X, Mt, M, affines)
    xv = (xv - MEAN.to(dev)) / STD.to(dev)
    out = []
    for copies in models:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type != "cpu"):
            o = copies[di](xv, Mt, M, theta)
        out.append(torch.sigmoid(o.float()).cpu().numpy()[0])
    return np.stack(out)


def rank_mean(P):
    """P (N, n_models, 12) probabilities (nan = study failed) -> (N, 12): per label, the mean over models of their
    percentile ranks across the test studies; a failed study gets the middle rank."""
    n = len(P)
    out = np.full((n, 12), 0.5)
    for c in range(12):
        ranks = []
        for mi in range(P.shape[1]):
            col = P[:, mi, c]
            ok = ~np.isnan(col)
            if ok.sum() < 2:
                continue
            r = np.full(n, 0.5)
            r[ok] = pd.Series(col[ok]).rank(pct=True).values
            ranks.append(r)
        if ranks:
            out[:, c] = np.mean(ranks, axis=0)
    return out


def write_submission(path, uids, P):
    sub = pd.DataFrame(rank_mean(P), columns=LABELS)
    sub.insert(0, "StudyInstanceUID", uids)
    sub.to_csv(path, index=False)


def _load(study_dir, series_df):
    try:
        return load_study(study_dir, series_df)
    except Exception as e:
        print("preprocessing failed", os.path.basename(study_dir), repr(e), flush=True)
        return {}


def main():
    t0 = time.time()
    comp = os.environ.get("COMP_DIR") or setting("RAW_DATA_DIR") or find_dir("test_series.csv")
    model_dir = os.environ.get("MODEL_DIR") or setting("MODEL_DIR") or find_model_dir(MODELS[0])
    print(f"weights from {model_dir}", flush=True)
    out_csv = os.environ.get("OUT") or (os.path.join(setting("SUBMISSION_DIR"), "submission.csv") if setting("SUBMISSION_DIR")
                                        else "submission.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    uids = pd.read_csv(f"{comp}/test.csv").StudyInstanceUID.astype(str).tolist()
    ser = pd.read_csv(f"{comp}/test_series.csv")
    ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str)
    N = len(uids)
    P = np.full((N, len(MODELS), 12), np.nan, np.float32)
    write_submission(out_csv, uids, P)   # a valid file from the start
    print(f"{N} test studies", flush=True)

    ncpu = max(1, min(4, os.cpu_count() or 4))
    pool = ProcessPoolExecutor(max_workers=ncpu, mp_context=mp.get_context("fork"))   # fork before CUDA starts
    list(pool.map(abs, range(ncpu)))
    devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    models = load_models(model_dir, devs)
    affines = [view_affines(torch.device(d)) for d in devs]
    print(f"{len(MODELS)} models x {len(devs)} devices ({time.time() - t0:.0f}s)", flush=True)

    work_q = queue.Queue(maxsize=8)

    def gpu_worker(di):
        while (item := work_q.get()) is not None:
            i, vols = item
            try:
                w = study_windows(vols)
                if w is not None:
                    P[i] = predict_study(models, di, affines[di], *w)
            except Exception as e:   # an uncaught error would stop this thread and block the queue
                print("study failed", uids[i], repr(e), flush=True)
                torch.cuda.empty_cache()

    threads = [threading.Thread(target=gpu_worker, args=(di,), daemon=True) for di in range(len(devs))]
    for t in threads:
        t.start()
    t_loop = time.time()
    with pool:
        ahead = 16
        futs = {i: pool.submit(_load, f"{comp}/test_series/{uids[i]}", ser[ser.StudyInstanceUID == uids[i]])
                for i in range(min(ahead, N))}
        nxt = len(futs)
        for i in range(N):
            if time.time() - t0 > TIME_BUDGET_S:
                print("time budget reached at study", i, flush=True)
                break
            vols = futs.pop(i).result()
            if nxt < N:
                futs[nxt] = pool.submit(_load, f"{comp}/test_series/{uids[nxt]}", ser[ser.StudyInstanceUID == uids[nxt]])
                nxt += 1
            if vols:
                work_q.put((i, vols))
            if i % 50 == 0:
                print(f"{i}/{N} {time.time() - t0:.0f}s ({(time.time() - t_loop) / (i + 1):.2f}s/study)", flush=True)
            if i % 200 == 0 and i:
                write_submission(out_csv, uids, P)
    for _ in threads:
        work_q.put(None)
    for t in threads:
        t.join()
    write_submission(out_csv, uids, P)
    print(f"done in {time.time() - t0:.0f}s | studies with predictions {int((~np.isnan(P[:, 0, 0])).sum())}/{N}", flush=True)


if __name__ == "__main__":
    main()
