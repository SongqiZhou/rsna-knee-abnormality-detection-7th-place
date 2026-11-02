"""Steps 8-9: the final label = 1/2 LLM label + 1/2 cross-model teacher.

Teacher: three 5-fold model families, each trained on a sibling LLM label (7a ConvNeXt-T, 7b EfficientNetV2-S with the
progressive window sampler, 7c EfficientNetV2-S with the fixed sampler). Each training study is scored by the one
fold model that did not see it (collect_oof.py -> WORK_DIR/oof/oof_teacher_*.csv). Per label, each family's
predictions -> percentile within the pool, mean of the three, and that order is rank-mapped onto the sorted values of
the LLM label (prevalence unchanged). Final = .5 LLM label + .5 mapped teacher, rounded to 4 decimals.
  file 8: pool = the 4349 training studies, LLM label = file 7.
  file 9 (--with-gold58): pool = the 4349 training studies + the 58 gold studies. The whole lineage of file 7 is redone
          over the pool (blend_reads.lineage with the gold58 reads 1g/2g/3g appended; read 1g gets the site shifts of
          step 4); gold58 teacher values = the mean of the five fold models (none of them trained on gold58). The host
          labels of the gold studies are not used anywhere.
usage: python -I labels/teacher_blend.py [--with-gold58]   -> LABEL_DIR/8_final_label.csv | 9_final_label_with_gold58_studies.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves the script's folder out
import common as C  # noqa: E402
import collect_oof  # noqa: E402
from blend_reads import lineage  # noqa: E402

A = .5
FAMILIES = ["oof_teacher_effnetv2s_progressive", "oof_teacher_effnetv2s_fixed", "oof_teacher_convnext_tiny"]
lgt = lambda p: np.log(p / (1 - p))
sig = lambda z: 1 / (1 + np.exp(-z))


def teacher_mix(llm, teachers):
    blend = sum(o.loc[llm.index].rank(pct=True) for o in teachers) / len(teachers)
    tb = llm.copy()
    for l in C.LABELS:
        srt = np.sort(llm[l].values)
        tb[l] = (1 - A) * llm[l].values + A * srt[blend[l].rank(method="first").values.astype(int) - 1]
    return tb.round(4)


def gold58_reads():
    """Reads 1-3 of the 58 gold studies in study_ids order; read 1 with the per-site Synovitis shifts of step 4
    (recovered from files 1 and 4 as the median logit difference per site)."""
    rd = lambda f: C.read_labels(C.settings.path("LABEL_DIR", f))[C.LABELS]
    g = list(C.study_ids("gold58").StudyInstanceUID)
    ti, im, sy = (rd(f).loc[g].astype(float) for f in ("1g_read_structured_gold58.csv", "2g_read_image_only_gold58.csv",
                                                         "3g_read_verification_gold58.csv"))
    f1, f4 = rd("1_read_structured.csv"), rd("4_read_structured_synovitis_fixed.csv")
    site = C.Cache().site()
    both = f4.index.intersection(f1.index).intersection(site.index)
    d = (lgt(f1.loc[both, "Synovitis"].clip(.005, .995)) - lgt(f4.loc[both, "Synovitis"].clip(1e-4, 1 - 1e-4))).groupby(site.loc[both]).median()
    d = d[d.abs() > 1e-3]
    ti["Synovitis"] = sig(lgt(ti.Synovitis.clip(.005, .995)) - site.reindex(ti.index).map(d).fillna(0).values)
    return ti, im, sy


def main():
    rd = lambda f: C.read_labels(C.settings.path("LABEL_DIR", f))[C.LABELS]
    oof = [collect_oof.load(f) for f in FAMILIES]
    if "--with-gold58" not in sys.argv:
        tb = teacher_mix(rd("7_llm_label.csv"), [o for o, _ in oof])
        out = "8_final_label.csv"
    else:
        gti, gim, gsy = gold58_reads()
        res = lineage(pd.concat([rd("4_read_structured_synovitis_fixed.csv"), gti]), pd.concat([rd("2_read_image_only.csv"), gim]),
                      pd.concat([rd("3_read_verification.csv"), gsy]))
        llm = res["LLM02b"]; llm.index.name = "StudyInstanceUID"
        tb = teacher_mix(llm, [pd.concat([o, g]) for o, g in oof])
        out = "9_final_label_with_gold58_studies.csv"
    tb.reset_index().to_csv(C.settings.path("LABEL_DIR", out), index=False)
    print(f"{len(tb)} studies -> {out}")


if __name__ == "__main__":
    main()
