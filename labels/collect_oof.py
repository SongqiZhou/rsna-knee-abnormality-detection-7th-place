"""Out-of-fold predictions of the 5-fold model families -> one CSV per family, the input of the Synovitis correction
(step 4) and of the teacher blend (steps 8-9).

Each fold model k was trained on the other four folds and evaluated on fold k and on the 58 gold studies (dense
evaluation by training/predict.py: CHECKPOINT_DIR/<run>/pred_denseT<T>.npz with arrays val, val_ids, gold, gold_ids).
Output rows: every training study with the prediction of the one model that did not train on it (fold 0-4), then the
58 gold studies with the mean of the five fold models (fold -1). Values are float32, written with 9 significant digits
so they read back exactly.
  oof_image_model                    CA0-CA4        T 1.25   trained on file 1 (for step 4)
  oof_teacher_effnetv2s_progressive  B16L3b_f0-f4   T 1.75   trained on file 7b
  oof_teacher_effnetv2s_fixed        S4bL4b_f0-f4   T 1.75   trained on file 7c
  oof_teacher_convnext_tiny          CNXT_L1b_f0-f4 T 1.25   trained on file 7a
usage: python -I labels/collect_oof.py [NAME ...]   (default: all four)   -> WORK_DIR/oof/NAME.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves the script's folder out
import common as C  # noqa: E402

FAMILIES = {"oof_image_model": ("1.25", "CA{}"), "oof_teacher_effnetv2s_progressive": ("1.75", "B16L3b_f{}"),
            "oof_teacher_effnetv2s_fixed": ("1.75", "S4bL4b_f{}"), "oof_teacher_convnext_tiny": ("1.25", "CNXT_L1b_f{}")}


def folds():
    return C.read_labels(os.path.join(C.settings.ROOT, "training", "folds.csv"))


def collect(name):
    T, run = FAMILIES[name]
    fo = folds(); parts, gold = [], []
    for k in range(5):
        z = np.load(C.settings.path("CHECKPOINT_DIR", run.format(k), f"pred_denseT{T}.npz"))
        ids = z["val_ids"].astype(str)
        assert set(ids) == set(fo.index[fo.fold == k]), f"{run.format(k)}: validation studies are not fold {k}"
        parts.append(pd.DataFrame(z["val"][:, :12], index=ids, columns=C.LABELS))
        gold.append(pd.DataFrame(z["gold"][:, :12], index=z["gold_ids"].astype(str), columns=C.LABELS))
    g0 = list(gold[0].index)
    gm = sum(g.loc[g0] for g in gold) / 5          # float32, summed in fold order
    out = pd.concat(parts + [gm])
    out.insert(0, "fold", [k for k, p in enumerate(parts) for _ in range(len(p))] + [-1] * len(gm))
    out.index.name = "StudyInstanceUID"
    return out


def load(name):
    """WORK_DIR/oof/NAME.csv -> (training studies, gold58 studies), float32 like the arrays they came from."""
    d = pd.read_csv(C.settings.path("WORK_DIR", "oof", f"{name}.csv"), dtype={"StudyInstanceUID": str, **{l: np.float32 for l in C.LABELS}})
    d = d.set_index("StudyInstanceUID")
    return d[d.fold >= 0][C.LABELS], d[d.fold < 0][C.LABELS]


if __name__ == "__main__":
    os.makedirs(C.settings.path("WORK_DIR", "oof"), exist_ok=True)
    for name in sys.argv[1:] or FAMILIES:
        f = C.settings.path("WORK_DIR", "oof", f"{name}.csv")
        collect(name).to_csv(f, float_format="%.9g")
        print("->", f)
