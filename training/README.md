# Training

| file | content |
|---|---|
| `train.py` | the trainer of every model; `--recipe` sets the hyper-parameters (table below) |
| `data.py` | window bags from the flat slice cache, the two window samplers, GPU augmentation |
| `model.py` | the 2.5D label-query network (training version of `inference/model.py`, same weight keys) |
| `predict.py` | out-of-fold predictions of a fold model (dense windows, at the inference attention temperature) |
| `package.py` | final weights -> `MODEL_DIR/<name>.pt` in the format `inference/` loads (strict-key check included) |
| `train_final.sh` | the 8 final models (`ONLY=final1`: the 4 of final submission 1) |
| `train_teachers.sh` | the 5-fold models the label pipeline needs (`STAGE=synovitis`, then `STAGE=teachers`) |
| `folds.csv` | the five folds of the 4,349 training studies (fold -1 = the 58 gold studies) |
| `monitor_labels_fold0_first100.csv` | report labels of the 100 monitoring studies of the two EfficientNetV2-S models of final submission 1 (see "Monitoring set") |

All paths come from `SETTINGS.json` (`CACHE_DIR` cache, `LABEL_DIR` labels, `RAW_DATA_DIR/train.csv` the host labels
of the 58 gold studies, which are used for monitoring only, `CHECKPOINT_DIR` runs, `MODEL_DIR` packaged models,
`WORK_DIR/logs` logs). Commands run from the repository root.

## Order

The default starts from the released labels (Kaggle dataset `songqizhou/rsna-knee-llm-labels`, files 1-9, in
`LABEL_DIR`) and only trains the final models:

```bash
GPUS="0 1 2 3" bash training/train_final.sh      # -> MODEL_DIR: 8 packaged models
```

Rebuilding the labels from the LLM reads (`labels/`) needs two rounds of fold models in between:

1. LLM reads -> files 1-3 (`labels/`)
2. `STAGE=synovitis bash training/train_teachers.sh` -> out-of-fold Effusion of CA0-CA4 -> files 4-7, 7a-7c (`labels/`)
3. `STAGE=teachers bash training/train_teachers.sh` -> out-of-fold predictions of the three teacher families -> files
   8 and 9 (`labels/`)
4. `bash training/train_final.sh`

## Final models

| packaged file | recipe | labels | training studies | seed | inference temperature |
|---|---|---|---|---|---|
| `effnetv2s_fixed.pt` | `effnet_fixed` | 8 | 4,349 (folds 0-4) | 3407 | 1.75 |
| `effnetv2s_progressive.pt` | `effnet_progressive` | 8 | 4,349 | 3407 | 1.75 |
| `convnext_small.pt` | `convnext`, ConvNeXt-S | 8 | 4,349 | 114514 | 1.25 |
| `convnext_tiny.pt` | `convnext`, ConvNeXt-T | 8 | 4,349 | 3407 | 1.25 |
| `effnetv2s_fixed_twin.pt` | `effnet_fixed` | 9 | 4,407 (folds -1..4) | 114514 | 1.75 |
| `effnetv2s_progressive_twin.pt` | `effnet_progressive` | 9 | 4,407 | 114514 | 1.75 |
| `convnext_small_twin.pt` | `convnext`, ConvNeXt-S | 9 | 4,407 | 3407 | 1.25 |
| `convnext_tiny_twin.pt` | `convnext`, ConvNeXt-T | 9 | 4,407 | 114514 | 1.25 |

Final submission 1 is the rank mean of the first four, final submission 2 of all eight. Labels 8 =
`8_final_label.csv`, 9 = `9_final_label_with_gold58_studies.csv` (the same recipe on the training set + the 58 gold
studies; the gold studies are trained on their LLM labels, never on the host labels).

## Fold models of the label pipeline

Fold model k trains on the other four folds; `predict.py` writes its predictions for fold k and the 58 gold studies.

| runs | recipe | labels | seed | temperature | used for |
|---|---|---|---|---|---|
| `CA0`-`CA4` | `effnet_c1` | 1 (structured read) | 42 | 1.25 | expected Synovitis rate per Effusion quintile (file 4) |
| `B16L3b_f0`-`f4` | `effnet_progressive` | 7b | 42 | 1.75 | teacher family 1 (files 8, 9) |
| `S4bL4b_f0`-`f4` | `effnet_fixed` | 7c | 42 | 1.75 | teacher family 2 |
| `CNXT_L1b_f0`-`f4` | `convnext`, ConvNeXt-T | 7a | 43 | 1.25 | teacher family 3 |

## Recipes

| | `effnet_fixed` | `effnet_progressive` | `convnext` | `effnet_c1` |
|---|---|---|---|---|
| backbone (timm, ImageNet weights) | `tf_efficientnetv2_s.in21k_ft_in1k` | same | `convnext_tiny.in12k_ft_in1k` / `convnext_small.in12k_ft_in1k` | same as fixed |
| token width | 384 | 384 | 512 | 384 |
| zoomed views | 13 | 13 | 13 | 4 general |
| training windows per study | 5 / 3 / 5 / 3 / 6 / 3 per series (sag FS / sag / cor FS / cor / ax FS / ax) | budget K = 36 (epochs 1-7), 75 (8-14), every centre (15-20); weights 6 : 2 : 3 : 1 : 3 : 1; 40% instance dropout | budget K = 50, weights 5 : 3 : 5 : 3 : 6 : 3 | as fixed |
| + per zoomed view | 4 (axial views), 3 (others) | same | same | same |
| backbone lr | 1e-4 | 1e-4 | 3e-5 | 1e-4 |
| drop path | 0.2 | 0.2 | 0.1 | 0.2 |
| lr schedule | 20 epochs | 20 epochs | 30 epochs, stopped after epoch 20 | 20 epochs |
| final weights | mean of the epoch-end weights of epochs 16-20, BatchNorm statistics re-estimated on 200 batches | same | mean of the epoch-end weights of epochs 16-20 | same as fixed |

Common to all: batch 4 studies; a window = 3 neighbouring slices at 384 px; window centres in the central 92% of each
series; soft-label BCE on each readout (query and MIL), averaged, with positive weight (1 - p) / p clipped to [1, 10]
(p = training prevalence of the label) and PF OA weighted x2; AdamW (head lr 5e-4, weight decay 0.05 on weight
matrices only), 5% linear warm-up then cosine; gradient norm clipped to 2; bf16 autocast; dropout 0.1 in the heads;
augmentation per (study, view): rotation +-15 degrees, field-of-view scale 0.85-1.15, shift +-8% (p 0.9), per-window
gain, contrast, gamma, low-resolution simulation and Rician noise, series dropout (p 0.1, at least 2 series and one
fat-suppressed kept), a 2-slice window step with p 0.2 (thicker slices).

**Monitoring set.** Every epoch the model is scored on the first `--val_limit` studies of fold `--val_fold` and on the
58 gold studies (host labels). With full-data training these studies are in the training set, so the numbers only
show that training runs; nothing is selected on them. In the two EfficientNetV2-S models of final submission 1
(`--ref_overrides 1`) the 100 monitoring studies were trained on their monitoring labels: as in my competition trainer,
the monitoring labels of a study replaced its training labels when the study was in both sets. Those labels are an
earlier report-only LLM read of the 100 studies, shipped as `monitor_labels_fold0_first100.csv` so the two models can be
retrained exactly as they were (the gold studies are not in their training set; `train.py` refuses `--ref_overrides` when they are). All other models train every study on the `--labels` file.

## Run time and hardware

One NVIDIA L40S 48 GB per model (the trainer uses one GPU).

| model | time |
|---|---|
| `effnet_fixed`, final (4,349 studies) | 6.6 h |
| `effnet_progressive`, final | 9.6 h |
| `effnet_c1`, one fold model (3,476 studies) | 3.2 h |
| `effnet_fixed`, one fold model | 5.6 h |
| `effnet_progressive`, one fold model | 6.8 h |
| `convnext`, ConvNeXt-T, final | about 9 h (estimate) |
| `convnext`, ConvNeXt-S, final | about 13 h (estimate) |
| `convnext`, ConvNeXt-T, one fold model | about 7 h (estimate) |

The EfficientNetV2-S times are the wall-clock times of the original runs. The ConvNeXt times are estimates from the
measured speed of this trainer on one L40S: 1.18 s per step for ConvNeXt-T and 1.82 s for ConvNeXt-S (1,087 steps per
epoch), plus one monitoring pass of 873 + 58 studies per epoch (3 and 4 min), i.e. 8.1 h and 12.3 h for 20 epochs,
rounded up because the same count (steps alone: 0.99 s x 1,087 x 20 = 6.0 h) falls short of the measured 6.6 h of
the EfficientNetV2-S fixed run. The twins train on 4,407 instead of 4,349 studies (1% more steps).

Out-of-fold prediction of one fold model (`predict.py`, 873 + 58 studies, dense): about 13 min.

## Checks

Run against the competition code on one L40S, with the released label files:

- **Model.** For a seed, `model.py` creates the same initial weights as the competition network (854 tensors for
  EfficientNetV2-S, 254 for ConvNeXt-T), its outputs in evaluation and training mode (dropout, drop path) are
  bit-identical, and the competition checkpoints load into it and into `inference/model.py` with strict keys.
- **Trainer, exact.** With deterministic GPU kernels (`torch.use_deterministic_algorithms`, math attention), 2 epochs
  of 15 steps of six configurations (the four recipes with the final labels, a twin on the training set + gold studies,
  a fold model with the 4 general zoomed views): every step's loss and gradient norm and the final weights (after the
  weight averaging and the BatchNorm re-estimation, or the mean of a stopped schedule) are bit-identical between
  `train.py` and the competition trainers.
- **Trainer, against the original runs.** In the normal (non-deterministic) GPU mode and with the exact arguments of the
  original runs, the running mean loss printed at steps 1 / 101 / 201 matches the original logs: final
  EfficientNetV2-S fixed 0.9878 / 1.1441 / 1.1333 (log 1.1332 at step 201), progressive 0.9811 / 1.1444 / 1.1341
  (identical), the fold models CA0 1.2279 / 1.1507 / 1.1316 (log 1.1315), S4bL4b_f0 1.2393 / 1.1883 / 1.1430 (log
  1.1884 at step 101), B16L3b_f0 1.2296 / 1.1879 / 1.1428 (identical). The first-step losses of the two final
  EfficientNetV2-S models are only reproduced with `--ref_overrides 1` and the 100 monitoring labels.
- **Backbones.** The weights timm downloads have the same SHA-256 as the files the runs started from
  (`convnext_tiny.in12k_ft_in1k` a1aefa40..., `convnext_small.in12k_ft_in1k` eb6f6171...).
- **Out-of-fold predictions.** `predict.py` on the original weights of two fold models (S4bL4b_f0 at T 1.75, CA0 at
  T 1.25) gives predictions for the 873 fold studies and the 58 gold studies that are bit-identical to the files the
  labels were built from.
- **Packaging.** `package.py` on the original final weights rebuilds all 8 files of the released weights dataset
  (same tensors, same config), and checkpoints written by `train.py` load into `inference/model.py` with strict keys.

## What to expect from a retrain

Training on GPUs is not bit-reproducible: two runs of the competition trainer with the same seed, data and flags give
the same first-step loss and then drift apart (after 30 steps their losses differ by up to 1.2e-3), so a retrain
follows the recipe but does not reproduce the weights. I did not retrain the final models with this repository, so I
have no leaderboard score of a retrain. The closest measurement is the second final submission: the four twins
(another seed, plus the 58 gold studies) as a 4-model ensemble scored the same as the four original models (0.970
public / 0.957 private).
