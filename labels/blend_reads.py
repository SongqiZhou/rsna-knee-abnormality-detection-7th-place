"""Steps 5-7: blend the three reads into the LLM label (files 5, 6, 7) and its sibling variants (7a, 7b, 7c).

Every blend is per label and is rank-mapped onto the marginal of file 4 (common.remap): the blend decides the order of
the studies, file 4 gives the values, so the prevalence of each label stays that of the Synovitis-corrected read 1.
rank = percentile within the pool (average ranks for ties).
  file 5  = .3 file4 + .2 file2 + .3 rank(file4) + .2 rank(file2)      (read 1 : read 2 = 60 : 40, probability + rank)
  LLM0k   = (1 - s) A + s S of file 5 (A) and file 3 (S), in one of three forms:
            p = probabilities, r = ranks, pr = .5 probability + .5 rank of each source
            LLM01 .4 pr | LLM02 .5 r | LLM03 .6 pr | LLM04 .6 p
  file 6  = LLM02 (1 : 1 rank mean of file 5 and file 3)
  LLM0kb  = LLM0k with Synovitis and Effusion taken from file 5 (the verification read over-calls those two)
  file 7 = LLM02b (the LLM label), 7a = LLM01b, 7b = LLM03b, 7c = LLM04b (the labels of the three teacher families)
Values are rounded to 4 decimals. lineage() is also used with the 58 gold studies added to the pool (file 9).
usage: python -I labels/blend_reads.py   -> LABEL_DIR/5_*.csv, 6_*.csv, 7_*.csv, 7a/7b/7c_*.csv"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves the script's folder out
import common as C  # noqa: E402

SPEC = {"LLM01": (.4, "pr"), "LLM02": (.5, "r"), "LLM03": (.6, "pr"), "LLM04": (.6, "p")}
SWAP = ["Synovitis", "Effusion"]
FILES = {"LLM02b": "7_llm_label.csv", "LLM01b": "7a_llm_label_variant.csv", "LLM03b": "7b_llm_label_variant.csv",
         "LLM04b": "7c_llm_label_variant.csv"}


def lineage(ti, im, sy):
    """ti = file 4, im = file 2, sy = file 3 (rows = the pool; ti's row order is the order of file 5, sy's that of the
    LLM labels) -> dict with file 5 (ar64), LLM01-04 and LLM01b-04b."""
    im = im.loc[ti.index, C.LABELS]
    rk = C.pct_rank
    res = {"ar64": C.remap(.3 * ti + .2 * im + .3 * rk(ti) + .2 * rk(im), ti).round(4)}
    t2, A, S = ti.loc[sy.index, C.LABELS], res["ar64"].loc[sy.index, C.LABELS], sy[C.LABELS]
    ra, rs = rk(A), rk(S)
    form = {"p": (A, S), "r": (ra, rs), "pr": (.5 * A + .5 * ra, .5 * S + .5 * rs)}
    for k, (s, f) in SPEC.items():
        a_, s_ = form[f]
        res[k] = C.remap((1 - s) * a_ + s * s_, t2).round(4)
        res[k + "b"] = res[k].copy()
        res[k + "b"][SWAP] = A[SWAP].values
    return res


def main():
    rd = lambda f: C.read_labels(C.settings.path("LABEL_DIR", f))[C.LABELS]
    res = lineage(rd("4_read_structured_synovitis_fixed.csv"), rd("2_read_image_only.csv"), rd("3_read_verification.csv"))
    out = {"5_blend_reads_1_2.csv": res["ar64"], "6_blend_reads_1_2_3.csv": res["LLM02"]} | {f: res[k] for k, f in FILES.items()}
    for f, d in out.items():
        d.reset_index().to_csv(C.settings.path("LABEL_DIR", f), index=False)
        print("->", f)


if __name__ == "__main__":
    main()
