"""Training cache, step 1: competition DICOMs -> one npz per study (CACHE_DIR/studies/<StudyInstanceUID>.npz).

Per study, one series per slot {SAG, COR, AX} x {FS, N} (FS = Fat_Suppression == 1 in train_series.csv; the longest
series if several), and for each series:
- every slice in geometric order (more than 48 slices: even subsampling to 48);
- in-plane orientation canonicalised from ImageOrientationPatient; right knees mirrored to the left-knee frame
  (coronal / axial: columns flipped, sagittal: slice order reversed), so medial / lateral sit consistently;
- a 140 mm physical centre crop (smaller fields of view are padded, never stretched), resized to 384 px;
- a per-series 1-99 percentile window -> uint8.
The npz also stores the study meta (laterality, sex, scanner "Manufacturer|Model|field", per-series spacing and
thickness) that the flat cache index and the Synovitis correction use. inference/preprocess.py is the same code for
the test layout.

DICOMs are read from the competition zip (as the original cache was built) or from the extracted train_series/ folder.
usage: python preprocess/prepare_data.py [--zip PATH] [--workers 48] [--limit N]
"""
import os, io, json, glob
import numpy as np
import pydicom
import cv2

CROP_MM = 140.0
OUT_PX = 384
MAX_SLICES = 48
PLANE_STD = {  # canonical (row, col) directions in LPS patient space
    "SAG": (np.array([0, 1, 0.]), np.array([0, 0, -1.])),
    "COR": (np.array([1, 0, 0.]), np.array([0, 0, -1.])),
    "AX": (np.array([1, 0, 0.]), np.array([0, 1, 0.])),
}
PLANE_AXIS = {"SAG": 0, "COR": 1, "AX": 2}
CSV_PLANE = {"Sagittal": "SAG", "Coronal": "COR", "Axial": "AX"}


def _hdr(src):
    return pydicom.dcmread(src, stop_before_pixels=True)


def _f(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def series_header(srcs, plane_hint=None):
    """srcs: list of path-or-bytes callables. Returns ordered entries + geometry."""
    hs = []
    for s in srcs:
        try:
            hs.append((s, _hdr(s())))
        except Exception:
            continue
    if not hs:
        return None
    d0 = hs[0][1]
    iop = getattr(d0, "ImageOrientationPatient", None)
    if iop is None:
        return None
    row, col = np.array(iop[:3], float), np.array(iop[3:], float)
    normal = np.cross(row, col)
    plane = plane_hint or ["SAG", "COR", "AX"][int(np.argmax(np.abs(normal)))]
    ax = PLANE_AXIS[plane]

    def pos(h):
        ipp = getattr(h, "ImagePositionPatient", None)
        if ipp is None:
            return _f(getattr(h, "InstanceNumber", 0))
        return float(np.array(ipp, float)[ax])   # ascending patient coordinate along plane axis
    hs.sort(key=lambda t: pos(t[1]))
    ps = getattr(d0, "PixelSpacing", [0.5, 0.5])
    # image-centre x (LPS, +x = patient left) for laterality fallback
    cxs = []
    for _, h in hs[:: max(1, len(hs) // 8)]:
        ipp, p = getattr(h, "ImagePositionPatient", None), getattr(h, "PixelSpacing", None)
        if ipp is None or p is None:
            continue
        r_, c_ = int(getattr(h, "Rows", 0)), int(getattr(h, "Columns", 0))
        cxs.append((np.array(ipp, float) + row * _f(p[1]) * c_ / 2 + col * _f(p[0]) * r_ / 2)[0])
    lat = str(getattr(d0, "Laterality", "") or getattr(d0, "ImageLaterality", "") or "").upper()
    return {"srcs": [s for s, _ in hs], "plane": plane, "row": row, "col": col, "n": len(hs),
            "spacing": (_f(ps[0], 0.5), _f(ps[1], 0.5)), "lat_tag": lat if lat in ("L", "R") else "",
            "cx": float(np.median(cxs)) if cxs else 0.0,
            "sex": str(getattr(d0, "PatientSex", "") or ""),
            "pid": str(getattr(d0, "PatientID", "") or ""),
            "scanner": "|".join(str(getattr(d0, k, "") or "") for k in
                                ("Manufacturer", "ManufacturerModelName", "MagneticFieldStrength")),
            "thick": _f(getattr(d0, "SliceThickness", 0)),
            "tr": _f(getattr(d0, "RepetitionTime", 0)), "te": _f(getattr(d0, "EchoTime", 0))}


def _pixels(src):
    d = pydicom.dcmread(src())
    px = d.pixel_array.astype(np.float32)
    if px.ndim == 3:
        px = px[px.shape[0] // 2]
    px = px * _f(getattr(d, "RescaleSlope", 1), 1) + _f(getattr(d, "RescaleIntercept", 0))
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        px = px.max() - px
    return px


def _crop_pad(px, sp, out_px):
    """Centre crop to CROP_MM (pad with min if FOV smaller), resize to out_px."""
    sy, sx = sp
    ch, cw = int(round(CROP_MM / sy)), int(round(CROP_MM / sx))
    h, w = px.shape
    canvas = np.full((max(h, ch), max(w, cw)), px.min(), np.float32)
    oy, ox = (canvas.shape[0] - h) // 2, (canvas.shape[1] - w) // 2
    canvas[oy:oy + h, ox:ox + w] = px
    H, W = canvas.shape
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    c = canvas[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(c, (out_px, out_px), interpolation=cv2.INTER_AREA)


def process_series(si, lat, out_px=OUT_PX, max_slices=MAX_SLICES):
    n = si["n"]
    idx = np.arange(n) if n <= max_slices else np.unique(np.linspace(0, n - 1, max_slices).round().astype(int))
    plane = si["plane"]
    r_std, c_std = PLANE_STD[plane]
    flip_c = float(np.dot(si["row"], r_std)) < 0
    flip_r = float(np.dot(si["col"], c_std)) < 0
    imgs = []
    for i in idx:
        try:
            imgs.append(_pixels(si["srcs"][i]))
        except Exception:
            imgs.append(None)
    good = [k for k, im in enumerate(imgs) if im is not None]
    if not good:
        return None
    for k in range(len(imgs)):
        if imgs[k] is None:
            imgs[k] = imgs[min(good, key=lambda j: abs(j - k))]
    out = []
    for im in imgs:
        if flip_c:
            im = im[:, ::-1]
        if flip_r:
            im = im[::-1, :]
        if lat == "R" and plane in ("COR", "AX"):
            im = im[:, ::-1]
        out.append(_crop_pad(np.ascontiguousarray(im), si["spacing"], out_px))
    vol = np.stack(out)
    if lat == "R" and plane == "SAG":
        vol = vol[::-1]
    sub = vol[:, ::4, ::4]
    p1, p99 = np.percentile(sub, [1, 99])
    vol = np.clip((vol - p1) / max(p99 - p1, 1e-3), 0, 1)
    return (vol * 255 + 0.5).astype(np.uint8)


def choose_slots(series_rows):
    """series_rows: list of (series_uid, plane, fluid, fs, si). Returns {slot: si}."""
    chosen = {}
    for uid, plane, fluid, fs, si in series_rows:
        if si is None:
            continue
        slot = f"{plane}_{'FS' if int(fs) == 1 else 'N'}"
        if slot not in chosen or si["n"] > chosen[slot]["n"]:
            si["series"] = uid
            chosen[slot] = si
    return chosen


def study_laterality(sis):
    tags = [s["lat_tag"] for s in sis if s["lat_tag"]]
    if tags:
        return max(set(tags), key=tags.count), "tag"
    cx = float(np.median([s["cx"] for s in sis]))
    if cx > 20:
        return "L", "cx"
    if cx < -20:
        return "R", "cx"
    return "L", "default"


def process_study(series_list, out_px=OUT_PX):
    """series_list: list of (series_uid, plane_csv, fluid, fs, [src callables]).
    Returns (arrays {slot: uint8[n,px,px]}, meta)."""
    rows = []
    for uid, plane_csv, fluid, fs, srcs in series_list:
        if len(srcs) < 3:
            continue
        si = series_header(srcs, CSV_PLANE.get(plane_csv))
        rows.append((uid, CSV_PLANE.get(plane_csv, si["plane"] if si else "AX"), fluid, fs, si))
    sis = [r[4] for r in rows if r[4] is not None]
    if not sis:
        return {}, {}
    lat, lat_src = study_laterality(sis)
    chosen = choose_slots(rows)
    arrays, meta = {}, {"lat": lat, "lat_src": lat_src, "sex": sis[0]["sex"], "pid": sis[0]["pid"],
                        "scanner": sis[0]["scanner"], "slots": {}}
    for slot, si in chosen.items():
        v = process_series(si, lat, out_px)
        if v is not None:
            arrays[slot] = v
            meta["slots"][slot] = {"series": si["series"], "n": si["n"], "kept": int(v.shape[0]),
                                   "spacing": si["spacing"], "thick": si["thick"], "tr": si["tr"], "te": si["te"]}
    return arrays, meta


# ----------------------------------------------------------------------------- sources
def dir_series_list(study_dir, series_df):
    """Kaggle test layout: study_dir/<series>/<sop>.dcm with series_df rows for this study."""
    out = []
    for r in series_df.itertuples():
        sd = os.path.join(study_dir, str(r.SeriesInstanceUID))
        files = sorted(glob.glob(os.path.join(sd, "*.dcm")))
        out.append((str(r.SeriesInstanceUID), r.Anatomical_Plane, int(r.Fluid_Sensitive), int(r.Fat_Suppression),
                    [(lambda f=f: f) for f in files]))
    return out


def zip_series_list(zf, names_by_series, study_uid, series_df, split="train_series"):
    out = []
    for r in series_df.itertuples():
        names = names_by_series.get((study_uid, str(r.SeriesInstanceUID)), [])
        blobs = [zf.read(n) for n in names]   # read once, sequential within the study
        out.append((str(r.SeriesInstanceUID), r.Anatomical_Plane, int(r.Fluid_Sensitive), int(r.Fat_Suppression),
                    [(lambda b=b: io.BytesIO(b)) for b in blobs]))
    return out


if __name__ == "__main__":
    import sys, zipfile, time, argparse
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from settings import path
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="", help="read the DICOMs from the competition zip instead of RAW_DATA_DIR/train_series")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="first N studies only (smoke test)")
    ap.add_argument("--uids", default="", help="file with one StudyInstanceUID per line (subset)")
    args = ap.parse_args()
    out_root = path("CACHE_DIR", "studies")
    os.makedirs(out_root, exist_ok=True)
    ser = pd.read_csv(path("RAW_DATA_DIR", "train_series.csv"))
    studies = sorted(ser.StudyInstanceUID.unique())
    if args.uids:
        keep = {l.strip() for l in open(args.uids) if l.strip()}
        studies = [s for s in studies if s in keep]
    if args.limit:
        studies = studies[:args.limit]
    todo = [s for s in studies if not os.path.exists(os.path.join(out_root, s + ".npz"))]
    print(f"{len(todo)} studies to process", flush=True)

    _zf = {}

    def series_list(st):
        if not args.zip:
            return dir_series_list(path("RAW_DATA_DIR", "train_series", st), ser[ser.StudyInstanceUID == st])
        if "zf" not in _zf:   # one zip handle and name table per worker process
            z = zipfile.ZipFile(args.zip)
            nb = {}
            for n in z.namelist():
                if n.startswith("train_series/") and n.endswith(".dcm"):
                    p = n.split("/")
                    nb.setdefault((p[1], p[2]), []).append(n)
            _zf["zf"], _zf["nb"] = z, nb
        return zip_series_list(_zf["zf"], _zf["nb"], st, ser[ser.StudyInstanceUID == st])

    def work(chunk):
        res = []
        for st in chunk:
            try:
                arrays, meta = process_study(series_list(st), OUT_PX)
                if arrays:
                    tmp = os.path.join(out_root, st + ".tmp.npz")
                    np.savez(tmp, meta=json.dumps(meta), **arrays)
                    os.replace(tmp, os.path.join(out_root, st + ".npz"))
                    res.append((st, "ok"))
                else:
                    res.append((st, "empty"))
            except Exception as e:
                res.append((st, f"ERR {type(e).__name__}: {e}"))
        return res

    chunks = [todo[i:i + 8] for i in range(0, len(todo), 8)]
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(work, chunks):
            for st, r in res:
                done += 1
                if r != "ok":
                    print("FAIL", st, r, flush=True)
            if done % 200 < 8:
                print(f"{done}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print("done", done, f"{time.time()-t0:.0f}s", flush=True)
