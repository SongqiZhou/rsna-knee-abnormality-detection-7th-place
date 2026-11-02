"""Out-of-fold predictions of a fold model (the inputs of the Synovitis correction and of the teacher labels).

Dense evaluation of CHECKPOINT_DIR/<tag>/last.pt on its held-out fold (--val_fold of its args.json, every study) and
on the 58 gold studies: a window at every centre of every series and zoomed view (stride 1), bf16, the backbone run
once per batch and the heads re-run for each attention temperature. Writes CHECKPOINT_DIR/<tag>/pred_denseT<T>.npz
with val (fold studies x 12 probabilities), val_ids, gold (58 x 12), gold_ids.
usage: python training/predict.py TAG [--T 1.75] [--stride 1]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import LAB, MEAN, STD, FOLDS, path, load_labels, parse_k, macro_auc  # noqa: E402
from model import KQNet  # noqa: E402
from data import FlatBags, collate, gpu_views, parse_rois, roi_tensor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--T", default="1.75", help="attention temperatures, comma list")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    e = ap.parse_args()
    out = path("CHECKPOINT_DIR", e.tag)
    a = argparse.Namespace(**json.load(open(f"{out}/args.json")))
    Ts = [float(t) for t in e.T.split(",")]
    dev = "cuda"
    folds = pd.read_csv(FOLDS, dtype={"StudyInstanceUID": str})
    tr_csv = pd.read_csv(path("RAW_DATA_DIR", "train.csv"), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    gold = tr_csv.loc[folds[folds.is_gold].StudyInstanceUID, LAB].astype(np.float32)
    ref = load_labels(a.ref_labels or a.labels)
    val_ids = [u for u in folds[folds.fold == a.val_fold].StudyInstanceUID if u in ref.index]
    gold_ids = list(gold.index)
    L = {u: ref.loc[u].values for u in val_ids}
    L.update({u: gold.loc[u].values for u in gold_ids})
    span = tuple(float(s) for s in a.span.split(","))
    rois = parse_rois(a.roi)

    def loader(ids):
        ds = FlatBags(ids, L, path("CACHE_DIR", "index.csv"), path("CACHE_DIR", "slices.npy"), parse_k(a.k), False,
                      eval_stride=e.stride, span=span, rois=rois)
        return DataLoader(ds, batch_size=1, num_workers=e.workers, collate_fn=collate, pin_memory=True)

    model = KQNet(a.arch, [r["plane"] for r in rois], D=a.D, drop=a.drop, drop_path=a.drop_path, pretrained=False,
                  grad_ckpt=False).to(dev)
    model.load_state_dict(torch.load(f"{out}/last.pt", map_location="cpu", weights_only=True))
    model.bb.to(memory_format=torch.channels_last)
    model.eval()
    mean, std = MEAN.to(dev), STD.to(dev)
    roi_t = roi_tensor(rois, dev)
    feats, cache = model._features, {}

    def cached(x):   # the backbone runs once per batch; the heads re-run per temperature
        if "f" not in cache:
            cache["f"] = feats(x)
        return cache["f"]
    model._features = cached

    res = {}
    for name, ids in (("val", val_ids), ("gold", gold_ids)):
        P = {t: [] for t in Ts}
        with torch.no_grad():
            for X, Mt, M, _, I in loader(ids):
                X, Mt, M = X.to(dev, non_blocking=True), Mt.to(dev), M.to(dev)
                xv, th = gpu_views(X, Mt, M, a.res, "none", rois=roi_t)
                xv = (xv - mean) / std
                cache.clear()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for t in Ts:
                        model.set_att_temp(t)
                        P[t].append(torch.sigmoid(model(xv, Mt, M, th).float()).cpu().numpy())
        Y = (np.stack([L[u] for u in ids]) >= 0.5).astype(int)
        for t in Ts:
            P[t] = np.concatenate(P[t])
            res[(name, t)] = (P[t], macro_auc(Y, P[t])[0])
    for t in Ts:
        np.savez(f"{out}/pred_denseT{t:g}.npz", val=res[("val", t)][0], gold=res[("gold", t)][0],
                 val_ids=np.array(val_ids), gold_ids=np.array(gold_ids))
        print(f"[{e.tag}] dense stride {e.stride} T {t:g}: fold {a.val_fold} {res[('val', t)][1]:.4f} (vs its "
              f"labels >= .5) | gold {res[('gold', t)][1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
