"""Read 1 (structured): GPT-6 astra reads each report together with 24 key slices and grades 24 attributes, then gives
a probability for each of the 12 labels.

- Images: three montages, 9 sagittal / 9 coronal slices over the central 15-85% of the series and 6 axial slices over
  10-90%, 256 px tiles (common.montage); fat-suppressed series when the study has one.
- System prompt: prompts/system_structured.txt (the shared labelling protocol + the attribute scales); user message:
  the montages and the report.
- One request per study, appended to WORK_DIR/labels/structured/rows.jsonl as it returns; a rerun resumes. While the
  endpoint refuses (quota), it is probed every 10 minutes.
- --parse writes LABEL_DIR/1_read_structured.csv: StudyInstanceUID, the 12 probabilities, the 24 grades.
- --training-labels writes WORK_DIR/labels/read1_labels.csv from file 1: StudyInstanceUID + the 12 probabilities in
  study_ids order, the training labels of the 5 fold models used by the Synovitis correction (labels/README.md).
- --gold58 reads the 58 gold studies instead (their reads enter file 9 only) -> 1g_read_structured_gold58.csv. The
  gold58 run used the same prompt; its user text lacked the words "(a missing plane is omitted)".
usage: API_URL=... API_KEY=... python -I labels/read_structured.py [--gold58] [--workers N]
       python -I labels/read_structured.py --parse [--gold58]
       python -I labels/read_structured.py --training-labels"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves the script's folder out
import common as C  # noqa: E402

SYSTEM = C.prompt("system_structured.txt")


def build(anon, uid, cache=None, rep=None):
    imgs = [b for b in (C.montage(cache, uid, anon, p) for p in ("sag", "cor", "ax")) if b is not None]
    note = "" if anon.startswith("g") else " (a missing plane is omitted)"
    return ([{"type": "text", "text": f"Study {anon}. Images: sagittal, coronal, axial montages{note}."}]
            + [C.image_part(b) for b in imgs] + [{"type": "text", "text": "Report:\n" + rep[uid]}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold58", action="store_true")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--training-labels", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    ids = C.study_ids("gold58" if args.gold58 else "train")
    rows_f = C.settings.path("WORK_DIR", "labels", "structured_gold58" if args.gold58 else "structured", "rows.jsonl")
    if args.training_labels:
        d = C.read_labels(C.settings.path("LABEL_DIR", "1_read_structured.csv")).loc[ids.StudyInstanceUID, C.LABELS]
        out = C.settings.path("WORK_DIR", "labels", "read1_labels.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True); d.to_csv(out)
        print(f"{len(d)} studies -> {out}")
    elif args.parse:
        d = C.parse_rows(rows_f, ids)[["StudyInstanceUID"] + C.PROBS + C.GRADES].rename(columns=dict(zip(C.PROBS, C.LABELS)))
        out = C.settings.path("LABEL_DIR", "1g_read_structured_gold58.csv" if args.gold58 else "1_read_structured.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True); d.to_csv(out, index=False)
        print(f"{len(d)}/{len(ids)} studies -> {out}")
    else:
        cache, rep = C.Cache(), C.reports()
        C.run_until_done(ids, SYSTEM, lambda a, u: build(a, u, cache, rep), rows_f, workers=args.workers, tries=4, wait=20,
                         timeout=600, check=False, abort=10)


if __name__ == "__main__":
    main()
