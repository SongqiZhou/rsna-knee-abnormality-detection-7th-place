"""DICOM -> one uint8 volume per series slot, the input format all four models were trained on.

- Series slot from the host CSV: {SAG, COR, AX} x {FS, N} (FS = Fat_Suppression == 1). If a study has two series
  for one slot, the one with more slices is kept.
- Slices in geometric order, at most 48 per series (even subsampling).
- In-plane orientation is canonicalised from ImageOrientationPatient, and right knees are mirrored into the
  left-knee frame (coronal / axial: flip columns; sagittal: reverse the slice order), so medial and lateral
  always sit on the same side.
- Fixed 140 mm crop around the image centre (smaller fields of view are padded, never stretched), resized to 384 px.
- Per-series 1-99 percentile window -> uint8.
"""
import os, glob
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


def _f(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def series_header(files, plane_hint=None):
    """Headers of one series -> its files sorted along the plane axis + geometry, or None."""
    hs = []
    for f in files:
        try:
            hs.append((f, pydicom.dcmread(f, stop_before_pixels=True)))
        except Exception:
            continue
    if not hs:
        return None
    d0 = hs[0][1]
    iop = getattr(d0, "ImageOrientationPatient", None)
    if iop is None:
        return None
    row, col = np.array(iop[:3], float), np.array(iop[3:], float)
    plane = plane_hint or ["SAG", "COR", "AX"][int(np.argmax(np.abs(np.cross(row, col))))]
    ax = PLANE_AXIS[plane]

    def pos(h):
        ipp = getattr(h, "ImagePositionPatient", None)
        if ipp is None:
            return _f(getattr(h, "InstanceNumber", 0))
        return float(np.array(ipp, float)[ax])   # ascending patient coordinate along the plane axis
    hs.sort(key=lambda t: pos(t[1]))
    ps = getattr(d0, "PixelSpacing", [0.5, 0.5])
    cxs = []   # x of the image centre (LPS, +x = patient left), the laterality fallback
    for _, h in hs[:: max(1, len(hs) // 8)]:
        ipp, p = getattr(h, "ImagePositionPatient", None), getattr(h, "PixelSpacing", None)
        if ipp is None or p is None:
            continue
        r_, c_ = int(getattr(h, "Rows", 0)), int(getattr(h, "Columns", 0))
        cxs.append((np.array(ipp, float) + row * _f(p[1]) * c_ / 2 + col * _f(p[0]) * r_ / 2)[0])
    lat = str(getattr(d0, "Laterality", "") or getattr(d0, "ImageLaterality", "") or "").upper()
    return {"files": [f for f, _ in hs], "plane": plane, "row": row, "col": col, "n": len(hs),
            "spacing": (_f(ps[0], 0.5), _f(ps[1], 0.5)), "lat_tag": lat if lat in ("L", "R") else "",
            "cx": float(np.median(cxs)) if cxs else 0.0, "thick": _f(getattr(d0, "SliceThickness", 0))}


def _pixels(f):
    d = pydicom.dcmread(f)
    px = d.pixel_array.astype(np.float32)
    if px.ndim == 3:
        px = px[px.shape[0] // 2]
    px = px * _f(getattr(d, "RescaleSlope", 1), 1) + _f(getattr(d, "RescaleIntercept", 0))
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        px = px.max() - px
    return px


def _crop_pad(px, sp, out_px):
    """Centre crop to CROP_MM (padded with the minimum if the field of view is smaller), resized to out_px."""
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
            imgs.append(_pixels(si["files"][i]))
        except Exception:
            imgs.append(None)
    good = [k for k, im in enumerate(imgs) if im is not None]
    if not good:
        return None
    for k in range(len(imgs)):   # an unreadable slice is replaced by its nearest readable neighbour
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


def choose_slots(rows):
    """rows [(plane, fat_suppression, header)] -> {slot: header}; the series with more slices wins a slot."""
    chosen = {}
    for plane, fs, si in rows:
        if si is None:
            continue
        slot = f"{plane}_{'FS' if int(fs) == 1 else 'N'}"
        if slot not in chosen or si["n"] > chosen[slot]["n"]:
            chosen[slot] = si
    return chosen


def study_laterality(sis):
    """DICOM laterality tag if any series has one, else the side of the image centre, else left."""
    tags = [s["lat_tag"] for s in sis if s["lat_tag"]]
    if tags:
        return max(set(tags), key=tags.count)
    cx = float(np.median([s["cx"] for s in sis]))
    if cx > 20:
        return "L"
    if cx < -20:
        return "R"
    return "L"


def load_study(study_dir, series_df, out_px=OUT_PX):
    """One test study (study_dir/<series>/<sop>.dcm, its rows of test_series.csv) ->
    {slot: (uint8 volume (n, out_px, out_px), number of slices in the series, slice thickness in mm)}."""
    rows = []
    for r in series_df.itertuples():
        files = sorted(glob.glob(os.path.join(study_dir, str(r.SeriesInstanceUID), "*.dcm")))
        if len(files) < 3:
            continue
        si = series_header(files, CSV_PLANE.get(r.Anatomical_Plane))
        rows.append((CSV_PLANE.get(r.Anatomical_Plane, si["plane"] if si else "AX"), int(r.Fat_Suppression), si))
    sis = [si for _, _, si in rows if si is not None]
    if not sis:
        return {}
    lat = study_laterality(sis)
    out = {}
    for slot, si in choose_slots(rows).items():
        v = process_series(si, lat, out_px)
        if v is not None:
            out[slot] = (v, si["n"], si["thick"])
    return out
