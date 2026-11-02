"""Training cache, step 2: the per-study npz files -> one flat uint8 slice array that training reads with mmap.

CACHE_DIR/slices.npy   (TOTAL, 384, 384) uint8, every stored slice of every series, series contiguous in slice
                       (geometric) order, studies in sorted StudyInstanceUID order
CACHE_DIR/index.csv    one row per (study, series slot): uid, slot, n, thick, n_orig, sp, lat, sex, scanner, off;
                       slice step in mm along the series normal ~= thick * n_orig / n (series with > 48 slices were
                       subsampled)
usage: python preprocess/build_cache.py [--workers 48]
"""
import os, sys, json, glob
import numpy as np, pandas as pd
import multiprocessing as mp

SLOTS = ["SAG_FS", "SAG_N", "COR_FS", "COR_N", "AX_FS", "AX_N"]


def scan(f):
    z = np.load(f)
    m = json.loads(str(z["meta"]))
    uid = os.path.basename(f)[:-4]
    rows = []
    for s in SLOTS:
        if s in z.files:
            sm = m["slots"].get(s, {})
            n = int(sm.get("kept", 0)) or None
            rows.append(dict(uid=uid, slot=s, n=n, thick=float(sm.get("thick", 0) or 0), n_orig=int(sm.get("n", 0) or 0),
                             sp=float((sm.get("spacing") or [0])[0] or 0), lat=m.get("lat", ""), sex=m.get("sex", ""),
                             scanner=m.get("scanner", "")))
    return f, rows


_MM = None


def _init(out):
    global _MM
    _MM = np.load(out, mmap_mode="r+")


def write(job):
    f, rows = job
    z = np.load(f)
    mm = _MM
    for r in rows:
        v = z[r["slot"]]
        assert v.shape[0] == r["n"] and v.shape[1:] == (384, 384), (f, r["slot"], v.shape, r["n"])
        mm[r["off"]:r["off"] + r["n"]] = v
    return len(rows)


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from settings import path
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    src, out_dir, nw = path("CACHE_DIR", "studies"), path("CACHE_DIR"), args.workers
    os.makedirs(out_dir, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(src, "*.npz")))
    ctx = mp.get_context("spawn")    # fork after pool threads deadlocked the writers
    with ctx.Pool(nw) as p:
        res = p.map(scan, fs, chunksize=8)
    # n from the meta can be missing; fall back to the stored array length
    fix = []
    for f, rows in res:
        if any(r["n"] is None for r in rows):
            z = np.load(f)
            for r in rows:
                r["n"] = int(z[r["slot"]].shape[0])
        fix.append((f, rows))
    off = 0
    for f, rows in fix:
        for r in rows:
            r["off"] = off; off += r["n"]
    print(f"[flat] {len(fix)} studies, {sum(len(r) for _, r in fix)} series, {off} slices, {off * 384 * 384 / 1e9:.1f} GB", flush=True)
    path = os.path.join(out_dir, "slices.npy")
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(off, 384, 384))
    del mm
    with ctx.Pool(nw, initializer=_init, initargs=(path,)) as p:
        done = 0
        for k in p.imap_unordered(write, fix, chunksize=4):
            done += k
            if done % 2000 < k:
                print(f"[flat] {done} series written", flush=True)
        _MM_main = np.load(path, mmap_mode="r+"); _MM_main.flush()
    pd.DataFrame([r for _, rows in fix for r in rows]).to_csv(os.path.join(out_dir, "index.csv"), index=False)
    print(f"[flat] wrote {done} series -> {path}", flush=True)
