"""Shared parts of the label pipeline: label names, the slice cache, image rendering for the LLM reads, the request
runner, and the rank helpers of the blends.

The three LLM reads see the same 140 mm / 384 px slices the models train on (the slice cache of preprocess/). Each
study is sent under an anonymous id (study_ids.csv: t*/v* = training studies, g* = the 58 gold studies), so no
StudyInstanceUID reaches the endpoint. The endpoint and key come from the environment (API_URL, API_KEY) only.
"""
import base64, io, json, os, re, sys, threading, time, urllib.request
import concurrent.futures as cf
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import settings  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
# reply keys: 24 integer grades (0-3, mcl_acute 0/1), then the 12 probabilities in LABELS order
GRADES = ["acl_grade", "mcl_grade", "mcl_acute", "mm_grade", "lm_grade", "mo_depth", "mo_area", "mo_ost", "mo_bml", "lo_depth",
          "lo_area", "lo_ost", "lo_bml", "pf_depth", "pf_area", "pf_ost", "pf_bml", "eff_size", "syn_thick", "hoffa",
          "baker_size", "edema_extent", "edema_pattern", "fracture"]
PROBS = ["p_ACL", "p_MCL", "p_MM", "p_LM", "p_MOA", "p_LOA", "p_PF", "p_EFF", "p_SYN", "p_BAK", "p_CON", "p_FRA"]
KEYS = GRADES + PROBS
# series per plane: fat-suppressed (fluid-sensitive) first, else the non-fat-suppressed one
PLANES = {"sag": ["SAG_FS", "SAG_N"], "cor": ["COR_FS", "COR_N"], "ax": ["AX_FS", "AX_N"]}
NAME = {"SAG_FS": "sagittal fat-suppressed (fluid-sensitive), anterior = left", "SAG_N": "sagittal non-fat-suppressed, anterior = left",
        "COR_FS": "coronal fat-suppressed (fluid-sensitive)", "COR_N": "coronal non-fat-suppressed",
        "AX_FS": "axial fat-suppressed (fluid-sensitive)", "AX_N": "axial non-fat-suppressed"}
ORIENT = {"sag": "slice numbers run MEDIAL (low) to LATERAL (high)", "cor": "MEDIAL = image left, LATERAL = image right",
          "ax": "anterior = top, MEDIAL = image left, LATERAL = image right"}


def prompt(name):
    return open(os.path.join(HERE, "prompts", name)).read()


def study_ids(which="train"):
    """StudyInstanceUID -> anonymous id, in the order the studies were read. which: train (4349), gold58 (58) or all."""
    m = pd.read_csv(os.path.join(HERE, "study_ids.csv"), dtype=str)
    if which == "train":
        m = m[~m.anon.str.startswith("g")]
    elif which == "gold58":
        m = m[m.anon.str.startswith("g")]
    return m.reset_index(drop=True)


def reports():
    tr = pd.read_csv(settings.path("RAW_DATA_DIR", "train.csv"), dtype={"StudyInstanceUID": str})
    return tr.set_index("StudyInstanceUID").Report


def read_labels(path):
    return pd.read_csv(path, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")


class Cache:
    """The slice cache: index.csv (one row per study and series slot) + slices.npy (all slices, 384 x 384 uint8)."""

    def __init__(self, d=None):
        d = d or settings.path("CACHE_DIR")
        self.ix = pd.read_csv(f"{d}/index.csv", dtype={"uid": str})
        self.by = self.ix.set_index(["uid", "slot"])
        self.sl = np.load(f"{d}/slices.npy", mmap_mode="r")

    def series(self, uid, plane):
        """-> (slot, row of index.csv) of the series used for this plane, or (None, None)."""
        for s in PLANES[plane]:
            if (uid, s) in self.by.index:
                r = self.by.loc[(uid, s)]
                return s, (r.iloc[0] if isinstance(r, pd.DataFrame) else r)
        return None, None

    def slice(self, row, k):
        return np.asarray(self.sl[int(row.off) + int(k)])

    def site(self):
        """Hospital proxy per study: scanner manufacturer (first word, upper case) + model, from the first series."""
        sc = self.ix.scanner.fillna("||").str.split("|", expand=True)
        s = (sc[0].str.split().str[0].str.upper().fillna("") + " " + sc[1].fillna("")).str.strip()
        return s.groupby(self.ix.uid).first()


def png(img):
    b = io.BytesIO(); img.save(b, format="PNG")
    return b.getvalue()


def montage(cache, uid, anon, plane):
    """Structured read: 9 sagittal / 9 coronal slices evenly over the central 15-85% of the series, 6 axial over
    10-90%, 256 px tiles in 3 columns, a title line and an "i/n" slice label per tile. -> PNG bytes or None."""
    slot, r = cache.series(uid, plane)
    if slot is None:
        return None
    lo, hi, n = (.10, .90, 6) if plane == "ax" else (.15, .85, 9)
    k = np.unique(np.round(np.linspace(lo, hi, n) * (r.n - 1)).astype(int)); T = 256; N = int(r.n)
    rows = 3 if plane == "cor" else int(np.ceil(len(k) / 3))   # the coronal canvas is always 3 x 3 tiles
    title = {"sag": f"{anon} sagittal {slot} (anterior = left), slice i/{N}",
             "cor": f"{anon} coronal {slot}, slice i/{N} (order as stored; check anatomy for medial/lateral)",
             "ax": f"{anon} axial {slot}, slice i/{N} (order as stored; check anatomy for direction)"}[plane]
    M = Image.new("L", (3 * T, rows * T + 18), 0); d = ImageDraw.Draw(M)
    d.text((4, 2), title, fill=255)
    for j, kk in enumerate(k):
        M.paste(Image.fromarray(cache.slice(r, kk)).resize((T, T), Image.BILINEAR), ((j % 3) * T, 18 + (j // 3) * T))
        d.text(((j % 3) * T + 4, 18 + (j // 3) * T + 4), f"{kk + 1}/{N}", fill=255)
    return png(M)


def frames(cache, uid, anon):
    """Image-only and verification reads: every slice of the sagittal, coronal and axial series as its own 384 px
    frame, plane by plane in slice order, with a title line (series, orientation) and an "i/n" slice label."""
    out = []
    for plane in ("sag", "cor", "ax"):
        slot, r = cache.series(uid, plane)
        if slot is None:
            continue
        N = int(r.n)
        for m in range(N):
            M = Image.new("L", (384, 384 + 18), 0); d = ImageDraw.Draw(M)
            d.text((4, 2), f"{anon} {plane} {m + 1}: {slot} {NAME[slot]}, {ORIENT[plane]}, slice i/{N}", fill=255)
            M.paste(Image.fromarray(cache.slice(r, m)).resize((384, 384), Image.BILINEAR), (0, 18))
            d.text((4, 18 + 4), f"{m + 1}/{N}", fill=255)
            out.append(png(M))
    return out


def image_part(b):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}}


def valid_reply(js):
    """Grades are integers in 0-3 (mcl_acute 0/1), probabilities in [0, 1]."""
    return all(float(js[c]) == int(float(js[c])) and 0 <= int(float(js[c])) <= (1 if c == "mcl_acute" else 3) for c in GRADES) \
        and all(0 <= float(js[c]) <= 1 for c in PROBS)


def done_ids(rows_f):
    ok = set()
    if os.path.exists(rows_f):
        for line in open(rows_f):
            try:
                r = json.loads(line)
                if "_error" not in r:
                    ok.add(r["id"])
            except Exception:
                pass
    return ok


def run_requests(todo, system, build, rows_f, model="gpt-6-astra", workers=1, tries=3, wait=15, timeout=300, check=True, abort=8):
    """Send one request per study and append each reply to rows_f (flush + fsync) as soon as it returns; studies that
    already have a successful row are skipped by the caller, so a rerun resumes. A study is retried `tries` times
    (sleep wait, 2 wait, ...); after `abort` consecutive failed studies (quota gone, endpoint down) the pass stops.
    system = the system prompt; build(anon, uid) -> the user message content. One worker by default: the endpoint
    failed under parallel load."""
    url, key = os.environ["API_URL"], os.environ["API_KEY"]
    lock, stop = threading.Lock(), threading.Event(); st = {"fail": 0, "ok": 0, "tok": 0}; t0 = time.time()

    def one(r):
        if stop.is_set():
            return None
        body = {"model": model, "max_tokens": 4000,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": build(r.anon, r.StudyInstanceUID)}]}
        err = ""
        for a in range(tries):
            if stop.is_set():
                return None
            try:
                res = json.load(urllib.request.urlopen(urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                headers={"Authorization": "Bearer " + key, "content-type": "application/json"}), timeout=timeout))
                js = json.loads(re.search(r"\{.*\}", res["choices"][0]["message"]["content"], re.S).group(0))
                if check and not valid_reply(js):
                    raise ValueError(f"invalid reply {json.dumps(js)[:150]}")
                return {"id": r.anon, "uid": r.StudyInstanceUID, **{c: js[c] for c in KEYS}, "_usage": res.get("usage"), "_t": time.time()}
            except Exception as e:
                err = f"{e} {getattr(e, 'read', lambda: b'')()[:200]!r}"; time.sleep(wait * (a + 1))
        return {"id": r.anon, "_error": err, "_t": time.time()}

    os.makedirs(os.path.dirname(rows_f), exist_ok=True)
    with open(rows_f, "a") as fo, cf.ThreadPoolExecutor(workers) as ex:
        for row in ex.map(one, list(todo.itertuples(index=False))):
            if row is None:
                continue
            with lock:
                fo.write(json.dumps(row) + "\n"); fo.flush(); os.fsync(fo.fileno())
                if "_error" in row:
                    st["fail"] += 1
                    if st["fail"] >= abort and not stop.is_set():
                        stop.set(); print(f"stopping after {abort} consecutive failures: {row['_error'][:200]}", flush=True)
                else:
                    st["fail"] = 0; st["ok"] += 1; st["tok"] += (row.get("_usage") or {}).get("total_tokens", 0)
                    if st["ok"] % 25 == 0:
                        print(f"{time.strftime('%m-%d %H:%M')} {st['ok']}/{len(todo)} done, {st['tok'] / 1e6:.2f} M tokens, "
                              f"{st['ok'] / max(1, time.time() - t0) * 3600:.0f} studies/h", flush=True)


def endpoint_up(model="gpt-6-astra"):
    body = {"model": model, "max_tokens": 5, "messages": [{"role": "user", "content": "Reply OK."}]}
    try:
        res = json.load(urllib.request.urlopen(urllib.request.Request(os.environ["API_URL"] + "/chat/completions", data=json.dumps(body).encode(),
                        headers={"Authorization": "Bearer " + os.environ["API_KEY"], "content-type": "application/json"}), timeout=90))
        return "choices" in res
    except Exception:
        return False


def run_until_done(ids, system, build, rows_f, poll_s=600, max_stalls=3, **kw):
    """Passes of run_requests until every study has a successful row. While the endpoint refuses (quota), it is probed
    every poll_s seconds; it gives up after max_stalls passes in a row without progress while the endpoint answers."""
    stalls = 0
    while len(done := done_ids(rows_f)) < len(ids):
        if not endpoint_up(kw.get("model", "gpt-6-astra")):
            print(f"{time.strftime('%m-%d %H:%M')} endpoint not available ({len(done)}/{len(ids)} done), next check in {poll_s // 60} min", flush=True)
            time.sleep(poll_s); continue
        run_requests(ids[~ids.anon.isin(done)], system, build, rows_f, **kw)
        stalls = stalls + 1 if len(done_ids(rows_f)) <= len(done) else 0
        if stalls >= max_stalls:
            print(f"giving up: {max_stalls} passes without progress ({len(done_ids(rows_f))}/{len(ids)})", flush=True)
            break
        time.sleep(30)


def parse_rows(rows_f, ids):
    """rows.jsonl -> one row per study (the last successful reply of each id), in the order the ids first succeeded;
    columns StudyInstanceUID + the 12 probabilities (named as LABELS) + the 24 grades."""
    ok = {}
    for line in open(rows_f):
        r = json.loads(line)
        if "_error" not in r:
            ok[r["id"]] = r
    d = pd.DataFrame(list(ok.values()))[["id"] + KEYS]
    d.insert(0, "StudyInstanceUID", d.id.map(dict(zip(ids.anon, ids.StudyInstanceUID))).values)
    return d.drop(columns="id")


# ---- rank helpers of the blends
def pct_rank(df):
    """Per label: rank / N within the rows of df (average ranks for ties)."""
    from scipy.stats import rankdata
    return df.apply(lambda c: pd.Series(rankdata(c) / len(c), index=c.index))


def remap(order, values):
    """Per label: keep the ordering of `order` (ties broken by row order) and take the sorted values of `values`, so
    the result has exactly the marginal (prevalence) of `values`."""
    out = order.copy()
    for l in LABELS:
        out[l] = np.sort(values[l].values)[order[l].rank(method="first").values.astype(int) - 1]
    return out
