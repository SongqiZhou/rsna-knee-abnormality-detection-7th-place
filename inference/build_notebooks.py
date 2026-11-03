"""Assemble the two Kaggle inference notebooks from inference/*.py: each module is shown in full in a %%writefile cell,
then infer.py runs. Final submission 1 (four models) -> kaggle/final1/, final submission 2 (eight models) ->
kaggle/final2/. Both load the weights from the public Kaggle Model (the weights dataset, with the same files, is attached too) and are pinned to the Kaggle image the published run used.
usage: python inference/build_notebooks.py   (then: kaggle kernels push -p inference/kaggle/final1)"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"
REPO = "https://github.com/SongqiZhou/rsna-knee-abnormality-detection-7th-place"

TABLE4 = """| Model | Backbone (timm) | Attention temperature | Public | Private |
|---|---|---|---|---|
| EfficientNetV2-S, fixed window sampler | `tf_efficientnetv2_s.in21k_ft_in1k` | 1.75 | 0.968 | 0.954 |
| EfficientNetV2-S, progressive window sampler | `tf_efficientnetv2_s.in21k_ft_in1k` | 1.75 | 0.969 | 0.954 |
| ConvNeXt-S | `convnext_small.in12k_ft_in1k` | 1.25 | 0.969 | **0.958** |
| ConvNeXt-T | `convnext_tiny.in12k_ft_in1k` | 1.25 | 0.969 | 0.957 |"""

STEPS = """1. **Preprocessing.** DICOM → up to 6 series (sagittal, coronal and axial, each with and without fat suppression), cropped to 140 mm at 384 px, right knees mirrored, at most 48 slices per series.
2. **Windows.** Every slice is the centre of a 3-slice window. 13 fixed zoomed views (menisci, patellofemoral joint, compartments) reuse the windows of their parent series.
3. **Network.** A shared 2D CNN turns each window into 12×12 tokens with a series embedding and a 3D position in a common knee frame. A label-query decoder and a gated attention-MIL head read every token of the study.
4. **Ensemble.** Per label, the mean of the models' percentile ranks."""

NB = {
    "final1": dict(
        slug="rsna-knee-7th-place-inference", title="RSNA Knee 7th Place Inference", ensemble="4",
        intro=f"""# RSNA Knee Abnormality Detection · 7th place inference

This notebook reproduces my final submission 1, the four-model rank mean: **private LB 0.957 (7th place), public LB 0.970**. [Final submission 2]({{nb8}}) adds a twin of each model and scores the same.

**No external training data.** The models were trained only on the competition images, with labels read from the competition reports and images by an LLM, starting from ImageNet-pretrained backbones. Weights: loaded from the Kaggle Model [RSNA Knee 7th Place](https://www.kaggle.com/models/songqizhou/rsna-knee-7th-place/PyTorch/default); the dataset [RSNA Knee 7th Place Models](https://www.kaggle.com/datasets/songqizhou/rsna-knee-7th-place-models) with the same files is attached too. Training code: [GitHub]({REPO}).

{TABLE4}
| **Rank mean of the four** | | | **0.970** | **0.957** |

All four models share one network and differ only in backbone and training recipe.

{STEPS}

On the hidden test set the notebook runs in about 2.5 h on 2×T4."""),
    "final2": dict(
        slug="rsna-knee-7th-place-inference-8-models", title="RSNA Knee 7th Place Inference 8 Models", ensemble="8",
        intro=f"""# RSNA Knee Abnormality Detection · 7th place inference, eight models

This notebook reproduces my final submission 2: the four models of [final submission 1]({{nb4}}) plus a twin of each, **private LB 0.957, public LB 0.970**. Each twin uses its model's recipe with another seed and is trained on the training set + the 58 gold studies (with my labels, not the host labels).

**No external training data.** The models were trained only on the competition images, with labels read from the competition reports and images by an LLM, starting from ImageNet-pretrained backbones. Weights: loaded from the Kaggle Model [RSNA Knee 7th Place](https://www.kaggle.com/models/songqizhou/rsna-knee-7th-place/PyTorch/default); the dataset [RSNA Knee 7th Place Models](https://www.kaggle.com/datasets/songqizhou/rsna-knee-7th-place-models) with the same files is attached too. Training code: [GitHub]({REPO}).

{TABLE4}
| **Rank mean of the four** (final submission 1) | | | **0.970** | **0.957** |
| **Rank mean of the eight** (this notebook) | | | **0.970** | **0.957** |

{STEPS}

On the hidden test set the notebook runs in about 5 h on 2×T4 (0.6 h per model), within the 9 h limit."""),
}
URL = {k: f"https://www.kaggle.com/code/songqizhou/{v['slug']}" for k, v in NB.items()}


def cell(i, kind, text):
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


for key, d in NB.items():
    run = "!python infer.py" if d["ensemble"] == "4" else "!ENSEMBLE=8 python infer.py"
    cells = [
        ("markdown", d["intro"].replace("{nb8}", URL["final2"]).replace("{nb4}", URL["final1"])),
        ("markdown", "## 1. Preprocessing\nDICOM → one uint8 volume per series slot."),
        ("code", "%%writefile preprocess.py\n" + open(f"{H}/preprocess.py").read().rstrip("\n")),
        ("markdown", "## 2. Network\nThe 2.5D label-query network shared by all models."),
        ("code", "%%writefile model.py\n" + open(f"{H}/model.py").read().rstrip("\n")),
        ("markdown", "## 3. Inference and rank mean\nCPU processes prepare the studies, one thread per GPU runs the models, and the per-label rank mean is written to `submission.csv`."),
        ("code", "%%writefile infer.py\n" + open(f"{H}/infer.py").read().rstrip("\n")),
        ("markdown", "## 4. Run"),
        ("code", 'import torch, timm\nprint("torch", torch.__version__, "| timm", timm.__version__, "| GPUs", torch.cuda.device_count())'),
        ("code", run),
        ("code", 'import pandas as pd\npd.read_csv("submission.csv").head()'),
    ]
    nb = {"cells": [cell(i, k, t) for i, (k, t) in enumerate(cells)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = f"{H}/kaggle/{key}"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(f"{out}/{d['slug']}.ipynb", "w"), indent=1)
    meta = {"id": f"songqizhou/{d['slug']}", "title": d["title"], "code_file": f"{d['slug']}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": "false", "enable_gpu": "true",
            "enable_internet": "false", "dataset_sources": ["songqizhou/rsna-knee-7th-place-models"],
            "model_sources": ["songqizhou/rsna-knee-7th-place/pyTorch/default/1"],
            "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [],
            "machine_shape": "NvidiaTeslaT4", "docker_image": IMAGE}
    json.dump(meta, open(f"{out}/kernel-metadata.json", "w"), indent=2)
    print(key, d["slug"], len(cells), "cells")
