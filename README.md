# RSNA Knee Abnormality Detection: 7th place solution

Code for my 7th-place solution (private 0.957 / public 0.970) in the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). The best single
model, a ConvNeXt-S, scored 0.969 public / 0.958 private.

**Pipeline.** The test set has no reports, so the models can only be as good as the labels read out of the training
reports. An LLM (GPT-6 Astra) reads every training study three times: the report with key slices, the images alone, and the
images with the report as a draft that may contain errors. The reads are rank-blended, a per-hospital correction fixes
Synovitis, and the result is averaged with the out-of-fold predictions of three model families. One 2.5D label-query
network (a shared 2D CNN over 3-slice windows of up to 6 series and 13 zoomed views, a label-query decoder and a gated
attention-MIL head) is trained in four variants, combined by a per-label rank mean. The models are trained on the
competition data only, starting from ImageNet-pretrained timm backbones.

| | public | private |
|---|---|---|
| final submission 1: rank mean of 4 models | 0.970 | 0.957 |
| final submission 2: the same 4 + 4 twins retrained with another seed and the 58 gold studies added (8 models) | 0.970 | 0.957 |
| best single model: ConvNeXt-S | 0.969 | 0.958 |

## Repository layout

| folder | content |
|---|---|
| `preprocess/` | competition DICOMs → the slice cache all models train on (`prepare_data.py`, `build_cache.py`) |
| `labels/` | the three LLM reads (prompts included), the Synovitis correction, the blends and the teacher blend; see `labels/README.md` |
| `training/` | training of every model (the 8 final models, and the fold models whose out-of-fold predictions feed the labels); see `training/README.md` |
| `inference/` | inference (`infer.py`) and the Kaggle notebooks built from it (`build_notebooks.py`, `kaggle/`) |
| `entry_points.md`, `SETTINGS.json`, `directory_structure.txt` | the commands from raw data to submission, the paths every script uses, and the folder layout |

## Quick start (inference with the released weights)

```bash
pip install -r requirements.txt
kaggle competitions download -c rsna-knee-abnormality-detection -p data && unzip -q data/rsna-knee-abnormality-detection.zip -d data/rsna-knee-abnormality-detection
kaggle datasets download songqizhou/rsna-knee-7th-place-models -p models --unzip
python inference/infer.py                 # final submission 1 -> submission/submission.csv
ENSEMBLE=8 python inference/infer.py      # final submission 2
```

On Kaggle, the public notebooks
[rsna-knee-7th-place-inference](https://www.kaggle.com/code/songqizhou/rsna-knee-7th-place-inference) (4 models) and
[rsna-knee-7th-place-inference-8-models](https://www.kaggle.com/code/songqizhou/rsna-knee-7th-place-inference-8-models)
run the same code with the released weights, published as the Kaggle Model
[songqizhou/rsna-knee-7th-place](https://www.kaggle.com/models/songqizhou/rsna-knee-7th-place/PyTorch/default) and, with the
same files, as the Kaggle dataset
[songqizhou/rsna-knee-7th-place-models](https://www.kaggle.com/datasets/songqizhou/rsna-knee-7th-place-models). All
commands, from the raw competition data to a submission, are in `entry_points.md`.

## Environment and run time

| | |
|---|---|
| hardware | every model trains on one NVIDIA GPU; checks and timings: Ubuntu 24.04.2, 2 × AMD EPYC 9554 (128 cores), 1 TB RAM, NVIDIA L40S (48 GB) |
| software | Python 3.12, PyTorch 2.13 with the CUDA 12.6 wheels (NVIDIA driver for CUDA ≥ 12.6), the exact versions in `requirements.txt` |
| Kaggle | the notebooks run offline on a pinned Kaggle image (`docker_image` in `inference/kaggle/*/kernel-metadata.json`; PyTorch 2.11, timm 1.0.29), 2 × T4 |
| training | one model in 7–13 h on one L40S (EfficientNetV2-S 6.6 / 9.6 h, ConvNeXt-T / -S about 9 / 13 h); the 8 final models run in parallel, one per GPU; details in `training/README.md` |
| inference | about 0.6 h per model on 2 × T4 for the ~1,300 hidden test studies: about 2.5 h for 4 models, about 5 h for 8 |
| labels | three LLM reads over the API, about 4, 7 and 8 h; the correction and the blends take seconds on a CPU |
| disk | competition data 265 GB (zip); slice cache 89 GB (per-study files) + 95 GB (flat array); checkpoints about 10 GB for the 8 final models, 25 GB with the fold models |

- **Network.** Training downloads the ImageNet-pretrained backbones through timm (Hugging Face Hub). The label reads
  need an OpenAI-compatible chat-completions endpoint (`API_URL`, `API_KEY` in the environment; never in a file).
  Inference runs offline.
- **Side effects.** The scripts write only to the folders in `SETTINGS.json` (`data/cache`, `data/labels`, `work`,
  `runs`, `models`, `submission` at the repository root), overwriting earlier outputs of the same step there. The
  competition data are only read.
- **Key assumptions.** All commands run from the repository root. The competition data (`train.csv`,
  `train_series.csv`, `train_series/`, `test*.csv`, `test_series/`, or the competition zip) are in `RAW_DATA_DIR`. A
  Linux machine with an NVIDIA GPU of 48 GB for training (the models train with batch 4, bf16 autocast and gradient
  checkpointing; smaller GPUs were not tested).

## Labels

The LLM replies are not bit-reproducible, so training starts by default from the released label files
([RSNA Knee LLM Labels](https://www.kaggle.com/datasets/songqizhou/rsna-knee-llm-labels), every intermediate step).
`labels/` regenerates all of them: from the saved replies and the out-of-fold predictions of the fold models, the
scripts reproduce every released file byte for byte. The hosts allowed LLM APIs for extracting labels from the
reports; the reads use the reports and the training images, and no study id other than an anonymous one leaves the
machine.

## Reproducing the scores

The inference code with the released weights reproduces both final submissions: on 32 local studies its per-model
probabilities and submission files are bitwise identical to the code that produced them, and on Kaggle both notebooks
give byte-identical outputs on the visible test studies; the four-model notebook scored 0.970 / 0.957 as a late
submission. Retraining follows the same recipes but not to the bit (GPU non-determinism); see `training/README.md`.

## Licence and acknowledgements

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (`LICENSE`), the winner licence of the
competition; the competition data are not included and remain under the MIRA license. Backbones from
[timm](https://github.com/huggingface/pytorch-image-models) (EfficientNetV2, ConvNeXt). The readouts follow
Query2Label / ML-Decoder (label queries) and Ilse et al. 2018 (gated attention-MIL); weight averaging follows SWA
(Izmailov et al. 2018). Thanks to RSNA, the contributing institutions and radiologists, and Kaggle.
