"""Step 4: per-hospital Synovitis correction of the structured read (file 1 -> file 4).

The structured read marks Synovitis almost exactly when the report uses the word, and how often reports use it depends
on the hospital, not on the knee. The scanner (manufacturer + model) stands in for the hospital. For every site with at
least 30 training studies, the site's Synovitis labels get one logit shift delta so that the site's mean label equals
its expected rate: each study's expected rate is the mean read-1 Synovitis label of the studies in the same quintile
of image Effusion, taken from an out-of-fold image model (5 fold models trained on file 1; collect_oof.py ->
WORK_DIR/oof/oof_image_model.csv). The order of studies within a site does not change; smaller sites and the other 11
labels are copied.
Also writes WORK_DIR/labels/synovitis_site_shifts.csv (site, studies, read-1 mean, expected mean, delta).
usage: python -I labels/synovitis_fix.py   -> LABEL_DIR/4_read_structured_synovitis_fixed.csv"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves the script's folder out
import common as C  # noqa: E402
import collect_oof  # noqa: E402

lgt = lambda p: np.log(p / (1 - p))
sig = lambda z: 1 / (1 + np.exp(-z))


def main():
    ids = C.study_ids("train")
    src = C.read_labels(C.settings.path("LABEL_DIR", "1_read_structured.csv")).loc[ids.StudyInstanceUID, C.LABELS]
    site = C.Cache().site()
    oof, _ = collect_oof.load("oof_image_model")
    T = src.index.intersection(oof.index).intersection(site.index)
    D = pd.DataFrame({"a": src.loc[T, "Synovitis"].clip(.005, .995), "e": oof.loc[T, "Effusion"], "site": site.loc[T]})
    D["q"] = pd.qcut(D.e, 5, labels=False)
    D["exp"] = D.groupby("q").a.transform("mean")
    out = src.copy(); cnt = D.site.value_counts(); rows = []
    for s in cnt[cnt >= 30].index:
        m = D.site == s; z = lgt(D.a[m].values); E = D.exp[m].sum()
        d = brentq(lambda x: sig(z - x).sum() - E, -10, 10)
        out.loc[D.index[m], "Synovitis"] = np.round(sig(z - d), 4)
        rows.append([s, int(m.sum()), D.a[m].mean(), E / m.sum(), d])
    shifts = pd.DataFrame(rows, columns=["site", "studies", "read1_mean", "expected_mean", "delta"]).sort_values("delta")
    os.makedirs(C.settings.path("WORK_DIR", "labels"), exist_ok=True)
    shifts.to_csv(C.settings.path("WORK_DIR", "labels", "synovitis_site_shifts.csv"), index=False)
    print(shifts.round(3).to_string(index=False))
    f = C.settings.path("LABEL_DIR", "4_read_structured_synovitis_fixed.csv")
    out.to_csv(f)
    print(f"{len(shifts)} sites shifted; Synovitis mean {src.Synovitis.mean():.3f} -> {out.Synovitis.mean():.3f}; -> {f}")


if __name__ == "__main__":
    main()
