"""Read 3 (verification): GPT-6 Astra gets every slice (as read 2) and the report, framed as a trainee's draft that may
contain errors, and decides each finding from the images.

- System prompt: prompts/system_verification.txt = the shared protocol, the attribute scales, the frame / orientation
  note, the trainee-draft framing, and typical report errors per label. The error rates come from the training set
  only (report-only labels against two image-based sources); the Synovitis paragraph states the per-hospital writing
  habit that labels/synovitis_fix.py corrects.
- User message: the frames, then "Trainee's draft report (may contain errors; reference only):" and the report.
- Checks, resume, quota probing as read_image_only.py: WORK_DIR/labels/verification/rows.jsonl; --parse ->
  LABEL_DIR/3_read_verification.csv; --gold58 -> 3g_read_verification_gold58.csv.
usage: API_URL=... API_KEY=... python -I labels/read_verification.py [--gold58] [--workers N]
       python -I labels/read_verification.py --parse [--gold58]"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves the script's folder out
import common as C  # noqa: E402

SYSTEM = C.prompt("system_verification.txt")


def build(anon, uid, cache=None, rep=None):
    return ([{"type": "text", "text": f"Study {anon}. Images (plane by plane, as described):"}]
            + [C.image_part(b) for b in C.frames(cache, uid, anon)]
            + [{"type": "text", "text": "Trainee's draft report (may contain errors; reference only):\n" + rep[uid]}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold58", action="store_true")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    ids = C.study_ids("gold58" if args.gold58 else "train")
    rows_f = C.settings.path("WORK_DIR", "labels", "verification_gold58" if args.gold58 else "verification", "rows.jsonl")
    if args.parse:
        d = C.parse_rows(rows_f, ids)[["StudyInstanceUID"] + C.PROBS].rename(columns=dict(zip(C.PROBS, C.LABELS)))
        out = C.settings.path("LABEL_DIR", "3g_read_verification_gold58.csv" if args.gold58 else "3_read_verification.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True); d.to_csv(out, index=False)
        print(f"{len(d)}/{len(ids)} studies -> {out}")
    else:
        cache, rep = C.Cache(), C.reports()
        C.run_until_done(ids, SYSTEM, lambda a, u: build(a, u, cache, rep), rows_f, workers=args.workers)


if __name__ == "__main__":
    main()
