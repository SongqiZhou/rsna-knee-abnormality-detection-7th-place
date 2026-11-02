"""Read 2 (image-only): GPT-6 astra reads every slice of the study, without the report, and gives the same 24 grades
and 12 probabilities as read 1.

- Images: every slice of the sagittal, coronal and axial series (fat-suppressed when the study has one) as its own
  384 px frame, plane by plane in slice order, each titled with its series and the standardised orientation
  (common.frames). About 92 frames per study.
- System prompt: prompts/system_image_only.txt (a self-contained image-only protocol with a reading checklist).
- Replies are checked (integer grades 0-3, probabilities in [0, 1]) and retried otherwise. Resume, quota probing and
  outputs as read_structured.py: WORK_DIR/labels/image_only/rows.jsonl; --parse -> LABEL_DIR/2_read_image_only.csv
  (StudyInstanceUID + the 12 probabilities); --gold58 -> 2g_read_image_only_gold58.csv.
usage: API_URL=... API_KEY=... python -I labels/read_image_only.py [--gold58] [--workers N]
       python -I labels/read_image_only.py --parse [--gold58]"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves the script's folder out
import common as C  # noqa: E402

SYSTEM = C.prompt("system_image_only.txt")


def build(anon, uid, cache=None):
    return [{"type": "text", "text": f"Study {anon}."}] + [C.image_part(b) for b in C.frames(cache, uid, anon)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold58", action="store_true")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    ids = C.study_ids("gold58" if args.gold58 else "train")
    rows_f = C.settings.path("WORK_DIR", "labels", "image_only_gold58" if args.gold58 else "image_only", "rows.jsonl")
    if args.parse:
        d = C.parse_rows(rows_f, ids)[["StudyInstanceUID"] + C.PROBS].rename(columns=dict(zip(C.PROBS, C.LABELS)))
        out = C.settings.path("LABEL_DIR", "2g_read_image_only_gold58.csv" if args.gold58 else "2_read_image_only.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True); d.to_csv(out, index=False)
        print(f"{len(d)}/{len(ids)} studies -> {out}")
    else:
        cache = C.Cache()
        C.run_until_done(ids, SYSTEM, lambda a, u: build(a, u, cache), rows_f, workers=args.workers)


if __name__ == "__main__":
    main()
