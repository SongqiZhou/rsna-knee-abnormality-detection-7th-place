"""Train one model of the solution (a final model, or a fold model whose out-of-fold predictions feed the labels).

A recipe (--recipe) sets every hyper-parameter; individual flags override it. The four recipes:
  effnet_fixed        EfficientNetV2-S, k windows per series (5/3/5/3/6/3 + 3-4 per zoomed view), 20 epochs, SWA of the
                      last 5 epoch-end weights + BatchNorm re-estimation
  effnet_progressive  the same with a per-study window budget K = 36 (epochs 1-7), 75 (8-14), every centre (15-20),
                      series weights fat-suppressed : other = 3 : 1 and sagittal : coronal : axial = 2 : 1 : 1, and 40%
                      instance dropout
  convnext            ConvNeXt (tiny or small, --arch), budget K = 50 split 5:3:5:3:6:3, lr schedule of 30 epochs stopped
                      after epoch 20, weights = mean of the epoch-end weights of epochs 16-20 (no BatchNorm)
  effnet_c1           effnet_fixed with the 4 general zoomed views only (the fold models of the Synovitis correction)

Loss: soft-label BCE per readout (query and MIL), positive weight (1 - prevalence) / prevalence clipped to [1, 10],
PF OA weighted x2, mean over the two readouts. AdamW (backbone lr, head lr 5e-4, wd 0.05 on matrices), 5% linear
warm-up then cosine, gradient norm clipped to 2, bf16 autocast, batch 4 studies.

Every epoch the model is scored on a monitoring set (the first --val_limit studies of fold --val_fold) and on the
58 gold studies; with full-data training both are in-sample, so these numbers only show that training runs. The
evaluation is kept because it is part of the random-number sequence of the original runs (each evaluation pass draws
one seed from the global generator).

Outputs in CHECKPOINT_DIR/<tag>/: args.json, hist.json, ckpt.pt (resume), last.pt (the final weights), result.json.
usage: python training/train.py --tag NAME --recipe RECIPE --labels FILE [--train_folds 0,1,2,3,4] [--seed N] [...]
"""
import os, sys, json, time, math, random, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings import path  # noqa: E402
from model import KQNet  # noqa: E402
from data import FlatBags, BudgetBags, SLOTS, collate, gpu_views, parse_rois, roi_tensor  # noqa: E402

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
FOLDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folds.csv")
# zoomed views 'SERIES:y0:y1:x0:x1:k' (fractions of the 140 mm slice, k training windows): 4 general views (axial,
# sagittal, two coronal halves), 5 menisci views, 4 patellofemoral views
ROI_GENERAL = "AX_FS:0:0.6:0.2:0.8:4;SAG_FS:0.1:0.7:0:0.6:3;COR_FS:0.2:0.8:0.4:1.0:3;COR_FS:0.2:0.8:0:0.6:3"
ROI_MENISCI = ("SAG_FS:0.38:0.83:0.1:0.55:3;SAG_FS:0.38:0.83:0.45:0.9:3;SAG_N:0.38:0.83:0.45:0.9:3;"
               "COR_FS:0.3:0.75:0.08:0.53:3;COR_FS:0.3:0.75:0.47:0.92:3")
ROI_PF = "AX_FS:0.02:0.5:0.2:0.8:4;AX_N:0.02:0.5:0.2:0.8:3;SAG_FS:0.12:0.62:0.08:0.58:3;SAG_N:0.12:0.62:0.08:0.58:3"
EFFNET = dict(arch="tf_efficientnetv2_s.in21k_ft_in1k", D=384, drop_path=0.2, bb_lr=1e-4, epochs=20, stop_epoch=0,
              roi=f"{ROI_GENERAL};{ROI_MENISCI};{ROI_PF}", sampler="fixed")
RECIPES = {
    "effnet_fixed": EFFNET,
    "effnet_progressive": {**EFFNET, "sampler": "budget", "budget_w": "6,2,3,1,3,1", "budget_sched": "36:7,75:7,ALL:6",
                           "inst_drop": 0.4},
    "convnext": dict(arch="convnext_tiny.in12k_ft_in1k", D=512, drop_path=0.1, bb_lr=3e-5, epochs=30, stop_epoch=20,
                     roi=f"{ROI_GENERAL};{ROI_MENISCI};{ROI_PF}", sampler="budget", budget_w="5,3,5,3,6,3",
                     budget_sched="50:20", inst_drop=0.0),
    "effnet_c1": {**EFFNET, "roi": ROI_GENERAL},
}


def label_path(p):
    return p if os.path.isabs(p) or os.path.exists(p) else path("LABEL_DIR", p)


def load_labels(p):
    d = pd.read_csv(label_path(p), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    return d[LAB].astype(np.float32)


def macro_auc(Y, P):
    per = {c: float(roc_auc_score(Y[:, j], P[:, j])) for j, c in enumerate(LAB) if len(np.unique(Y[:, j])) > 1}
    return float(np.mean(list(per.values()))), per


def parse_k(spec):
    return {x.split("=")[0]: int(x.split("=")[1]) for x in spec.split(",") if x}


def parse_sched(spec):
    return [(k, int(e)) for k, e in (s.split(":") for s in spec.split(","))]


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--labels", required=True, help="training labels CSV (StudyInstanceUID + 12 soft labels); a bare "
                    "file name is read from LABEL_DIR")
    ap.add_argument("--ref_labels", default="", help="labels of the monitoring set ('' = --labels)")
    ap.add_argument("--ref_overrides", type=int, default=0,
                    help="1: a monitoring study that is also a training study is trained on its --ref_labels value "
                         "(and a gold study on its host label), as in the competition trainer; the two EfficientNetV2-S "
                         "models of final submission 1 were trained this way (100 fold-0 studies)")
    ap.add_argument("--train_folds", default="0,1,2,3,4", help="folds of training/folds.csv; -1 = the 58 gold studies")
    ap.add_argument("--val_fold", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0, help="monitoring set: first N studies of --val_fold (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch"); ap.add_argument("--D", type=int); ap.add_argument("--drop_path", type=float)
    ap.add_argument("--bb_lr", type=float); ap.add_argument("--roi")
    ap.add_argument("--epochs", type=int, help="length of the lr schedule")
    ap.add_argument("--stop_epoch", type=int, help="stop after this many epochs (0 = --epochs); the final weights are "
                    "then the mean of the epoch-end weights of the last --swa epochs before the stop")
    ap.add_argument("--sampler", choices=["fixed", "budget"])
    ap.add_argument("--budget_w", help="budget sampler: weights of SAG_FS,SAG_N,COR_FS,COR_N,AX_FS,AX_N")
    ap.add_argument("--budget_sched", help="budget sampler: 'K:epochs,...' (K an int or ALL)")
    ap.add_argument("--inst_drop", type=float, help="budget sampler: instance dropout of the built windows")
    ap.add_argument("--k", default="SAG_FS=5,SAG_N=3,COR_FS=5,COR_N=3,AX_FS=6,AX_N=3", help="fixed sampler: windows per series")
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--span", default="0.04,0.96", help="window centres cover this central part of each series")
    ap.add_argument("--eval_stride", type=int, default=3, help="per-epoch monitoring: every n-th window centre")
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps with linear warm-up")
    ap.add_argument("--swa", type=int, default=5, help="average of the epoch-end weights of the last N epochs")
    ap.add_argument("--swa_bn_steps", type=int, default=200, help="training batches to re-estimate BatchNorm statistics")
    ap.add_argument("--slot_drop", type=float, default=0.1, help="p of dropping a series from a training bag")
    ap.add_argument("--thick_p", type=float, default=0.2, help="p of a 2-slice step (thicker-slice simulation)")
    ap.add_argument("--label_w", default="PF OA=2", help="per-label loss weights 'Label=w,...' (others 1.0)")
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test: steps per epoch")
    ap.add_argument("--resume", type=int, default=1)
    a = ap.parse_args(argv)
    for k, v in RECIPES[a.recipe].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    a.stop_epoch = a.stop_epoch or a.epochs
    return a


def main(argv=None):
    a = get_args(argv)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"
    torch.backends.cudnn.benchmark = False   # window counts vary per batch: benchmark would re-tune every new shape
    out = path("CHECKPOINT_DIR", a.tag); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(f"{out}/args.json", "w"), indent=1)

    folds = pd.read_csv(FOLDS, dtype={"StudyInstanceUID": str})
    tr_csv = pd.read_csv(path("RAW_DATA_DIR", "train.csv"), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    gold = tr_csv.loc[folds[folds.is_gold].StudyInstanceUID, LAB].astype(np.float32)   # host labels, monitoring only
    lab = load_labels(a.labels)
    ref = load_labels(a.ref_labels) if a.ref_labels else lab
    tf = [int(f) for f in a.train_folds.split(",")]
    train_ids = [u for u in folds[folds.fold.isin(tf)].StudyInstanceUID if u in lab.index]
    val_ids = [u for u in folds[folds.fold == a.val_fold].StudyInstanceUID if u in ref.index]
    if a.val_limit:
        val_ids = val_ids[:a.val_limit]
    gold_ids = list(gold.index)
    L = {u: lab.loc[u].values for u in train_ids}   # training targets
    LE = {u: ref.loc[u].values for u in val_ids}    # monitoring targets
    LE.update({u: gold.loc[u].values for u in gold_ids})
    if a.ref_overrides:
        L.update(LE)
    span = tuple(float(s) for s in a.span.split(","))
    ix, flat = path("CACHE_DIR", "index.csv"), path("CACHE_DIR", "slices.npy")
    k_tr = parse_k(a.k)
    rois = parse_rois(a.roi)
    if a.sampler == "budget":
        w = dict(zip(SLOTS, [float(x) for x in a.budget_w.split(",")]))
        ds_tr = BudgetBags(train_ids, L, ix, flat, k_tr, True, span=span, slot_drop=a.slot_drop, thick_p=a.thick_p,
                           rois=rois, weights=w, schedule=parse_sched(a.budget_sched), drop=a.inst_drop)
    else:
        ds_tr = FlatBags(train_ids, L, ix, flat, k_tr, True, span=span, slot_drop=a.slot_drop, thick_p=a.thick_p, rois=rois)
    mk_ev = lambda ids, st: FlatBags(ids, {**L, **LE}, ix, flat, k_tr, False, eval_stride=st, span=span, rois=rois)
    tl = DataLoader(ds_tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True,
                    pin_memory=True, persistent_workers=True, prefetch_factor=4)
    ev_loader = lambda ids, st: DataLoader(mk_ev(ids, st), batch_size=2, num_workers=a.workers, collate_fn=collate,
                                           pin_memory=True)
    vl, gl = ev_loader(val_ids, a.eval_stride), ev_loader(gold_ids, a.eval_stride)
    n_gold_tr = len(set(train_ids) & set(gold_ids))
    assert not (a.ref_overrides and n_gold_tr), "--ref_overrides would train the gold studies on their host labels"
    print(f"[{a.tag}] recipe {a.recipe} | train {len(train_ids)} (gold {n_gold_tr}) | monitor {len(val_ids)} + gold "
          f"{len(gold_ids)} | labels {a.labels} | {a.arch}@{a.res} | sampler {a.sampler} | seed {a.seed}", flush=True)

    model = KQNet(a.arch, [r["plane"] for r in rois], D=a.D, drop=a.drop, drop_path=a.drop_path,
                  grad_ckpt=bool(a.grad_ckpt)).to(dev)
    model.bb.to(memory_format=torch.channels_last)
    # parameter groups: backbone and head, each split into decayed matrices and undecayed vectors / embeddings
    bb_named = [(n, p) for n, p in model.named_parameters() if n.startswith("bb.")]
    groups = {}
    for n, p in bb_named:
        groups.setdefault(p.ndim <= 1, []).append(p)
    pg = [{"params": ps, "lr": a.bb_lr, "weight_decay": 0.0 if nd else a.wd} for nd, ps in groups.items()]
    head_p = [(n, p) for n, p in model.named_parameters() if not n.startswith("bb.")]
    pg += [{"params": [p for n, p in head_p if p.ndim > 1 and "emb" not in n and n != "queries"], "lr": a.head_lr, "weight_decay": a.wd},
           {"params": [p for n, p in head_p if p.ndim <= 1 or "emb" in n or n == "queries"], "lr": a.head_lr, "weight_decay": 0.0}]
    for g in pg:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(pg, betas=(0.9, 0.999))
    spe = a.max_steps or len(tl)
    total = spe * a.epochs
    wu = max(1, int(a.warmup * total))
    lr_at = lambda s: (s + 1) / wu if s < wu else 0.5 * (1 + math.cos(math.pi * (s - wu) / max(1, total - wu)))

    Ytr = np.stack([L[u] for u in train_ids])
    prev = np.clip(Ytr.mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
    lw = torch.ones(len(LAB), device=dev)
    for item in [x for x in a.label_w.split(",") if x]:
        name, wt = item.rsplit("=", 1); lw[LAB.index(name)] = float(wt)
    _bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    bce = lambda logit, y: (_bce(logit, y) * lw).mean()
    Yv = (np.stack([LE[u] for u in val_ids]) >= 0.5).astype(int)
    Yg = np.stack([LE[u] for u in gold_ids]).astype(int)
    mean, std = MEAN.to(dev), STD.to(dev)
    roi_t = roi_tensor(rois, dev)

    def run_eval(m, loader):
        m.eval(); P = []
        with torch.no_grad():
            for X, Mt, M, _, I in loader:
                X, Mt, M = X.to(dev, non_blocking=True), Mt.to(dev), M.to(dev)
                xv, th = gpu_views(X, Mt, M, a.res, "none", rois=roi_t)
                xv = (xv - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    P.append(torch.sigmoid(m(xv, Mt, M, th).float()).cpu().numpy())
        return np.concatenate(P)

    def bn_recalibrate(m, steps):
        """Re-estimate BatchNorm running statistics of averaged weights (cumulative average over training batches)."""
        bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
        if not bns:
            return 0
        mom = [x.momentum for x in bns]
        for x in bns:
            x.reset_running_stats(); x.momentum = None
        m.train(); n = 0
        with torch.no_grad():
            for X, Mt, M, Y, I in tl:
                if n >= steps:
                    break
                X, Mt, M = X.to(dev, non_blocking=True), Mt.to(dev), M.to(dev)
                xv, th = gpu_views(X, Mt, M, a.res, "strong", rois=roi_t)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    m((xv - mean) / std, Mt, M, th)
                n += 1
        for x, mo in zip(bns, mom):
            x.momentum = mo
        return len(bns)

    hist, step, start, t0 = [], 0, 0, time.time()
    swa_state, swa_n, snaps = None, 0, {}
    swa_from = a.stop_epoch - a.swa   # epochs swa_from .. stop_epoch - 1 (0-based) are averaged
    ck = f"{out}/ckpt.pt"
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        hist, start, step = s["hist"], s["ep"] + 1, s["step"]
        swa_state, swa_n, snaps = s.get("swa"), s.get("swa_n", 0), s.get("snaps", {})
        swa_state = {k: v.to(dev) for k, v in swa_state.items()} if swa_state is not None else None
        if isinstance(ds_tr, BudgetBags):   # the K schedule reads the epoch from the sample counter
            ds_tr.count.value = start * len(ds_tr.uids)
        print(f"[{a.tag}] resumed after epoch {s['ep']}", flush=True)
    for ep in range(start, a.stop_epoch):
        model.train(); tot, nb, te = 0.0, 0, time.time()
        gsum = 0.0
        for it, (X, Mt, M, Y, I) in enumerate(tl):
            if a.max_steps and it >= a.max_steps:
                break
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_at(step)
            X, Mt, M, Y = X.to(dev, non_blocking=True), Mt.to(dev), M.to(dev), Y.to(dev)
            xv, th = gpu_views(X, Mt, M, a.res, "strong", rois=roi_t)
            xv = ((xv - mean) / std)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xv, Mt, M, th, all_heads=True)
            loss = sum(bce(lg.float(), Y) for lg in logits) / len(logits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gsum += float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))
            opt.step(); step += 1
            tot += loss.item(); nb += 1
            if it % 100 == 0:
                print(f"[{a.tag}] ep{ep} it{it}/{spe} loss {tot/nb:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"({time.time()-te:.0f}s)", flush=True)
        Pv, Pg = run_eval(model, vl), run_eval(model, gl)
        va, vau = macro_auc(Yv, Pv); ga, gau = macro_auc(Yg, Pg)
        hist.append({"ep": ep, "loss": tot / max(nb, 1), "monitor": va, "gold": ga, "monitor_per": vau, "gold_per": gau,
                     "gnorm": gsum / max(nb, 1), "t_train": time.time() - te, "t": time.time() - t0})
        print(f"[{a.tag}] ep{ep} loss {tot/max(nb,1):.4f} | monitor {va:.4f} | gold {ga:.4f} (in-sample when trained on) | "
              f"|g| {gsum/max(nb,1):.2f} | {time.time()-te:.0f}s ep, {time.time()-t0:.0f}s total", flush=True)
        json.dump(hist, open(f"{out}/hist.json", "w"), indent=1)
        if a.swa and ep >= swa_from:
            sd = model.state_dict()
            if a.stop_epoch < a.epochs:   # stopped schedule: keep the epoch-end weights, averaged in float64 at the stop
                snaps[ep] = {k: v.detach().cpu().clone() for k, v in sd.items()}
            elif swa_state is None:       # full schedule: running float32 mean
                swa_state = {k: (v.detach().float().clone() if v.is_floating_point() else v.clone()) for k, v in sd.items()}
                swa_n = 1
            else:
                swa_n += 1
                for k, v in sd.items():
                    if v.is_floating_point():
                        swa_state[k] += (v.detach().float() - swa_state[k]) / swa_n
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ep": ep, "step": step, "hist": hist,
                    "swa": swa_state, "swa_n": swa_n, "snaps": snaps}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
    res = {}
    if a.swa and snaps:
        sds = [snaps[e] for e in sorted(snaps)]
        avg = {k: ((sum(sd[k].double() for sd in sds) / len(sds)).to(v.dtype) if v.is_floating_point() else sds[-1][k])
               for k, v in sds[-1].items()}
        assert not any(isinstance(x, nn.modules.batchnorm._BatchNorm) for x in model.modules()), \
            "a stopped schedule averages without BatchNorm re-estimation"
        model.load_state_dict(avg)
        res["swa_epochs"] = [e + 1 for e in sorted(snaps)]
        print(f"[{a.tag}] final weights: mean of epochs {res['swa_epochs']}", flush=True)
    elif a.swa and swa_state is not None:
        ref_sd = model.state_dict()
        model.load_state_dict({k: v.to(ref_sd[k].dtype) for k, v in swa_state.items()})
        nbn = bn_recalibrate(model, a.swa_bn_steps)
        res["swa_n"], res["swa_bn_layers"] = swa_n, nbn
        print(f"[{a.tag}] final weights: SWA of the last {swa_n} epochs (BatchNorm layers re-estimated: {nbn})", flush=True)
    torch.save(model.state_dict(), f"{out}/last.pt")
    res["t"] = time.time() - t0
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    print(f"[{a.tag}] done {res['t']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
