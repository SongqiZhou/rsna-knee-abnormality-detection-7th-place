# Entry points

All commands run from the repository root. The paths they read and write are set only in `SETTINGS.json` (read by
`settings.py`); the folder layout is in `directory_structure.txt`, hardware and run time in `README.md`.

## 1. Environment

```bash
pip install -r requirements.txt
```

## 2. Data

```bash
# competition data -> RAW_DATA_DIR (data/rsna-knee-abnormality-detection)
kaggle competitions download -c rsna-knee-abnormality-detection -p data
unzip -q data/rsna-knee-abnormality-detection.zip -d data/rsna-knee-abnormality-detection

# the released labels -> LABEL_DIR (data/labels): the default training targets (step 4 rebuilds them instead)
kaggle datasets download songqizhou/rsna-knee-llm-labels -p data/labels --unzip
```

## 3. Preprocess (prepare_data)

```bash
python preprocess/prepare_data.py      # DICOM -> CACHE_DIR/studies/<StudyInstanceUID>.npz   (--zip PATH reads the zip directly)
python preprocess/build_cache.py       # -> CACHE_DIR/slices.npy + CACHE_DIR/index.csv, the cache training reads
```

## 4. Labels from scratch (optional)

Only needed to rebuild the files of step 2. The three reads call an LLM endpoint (`API_URL`, `API_KEY` in the
environment); the rest runs from their saved replies. Full commands and checks: `labels/README.md`.

```bash
python -I labels/read_structured.py && python -I labels/read_structured.py --parse          # read 1
python -I labels/read_image_only.py && python -I labels/read_image_only.py --parse          # read 2
python -I labels/read_verification.py && python -I labels/read_verification.py --parse      # read 3
python -I labels/read_structured.py --training-labels
STAGE=synovitis bash training/train_teachers.sh                 # 5 fold models -> out-of-fold Effusion
python -I labels/collect_oof.py oof_image_model && python -I labels/synovitis_fix.py        # file 4
python -I labels/blend_reads.py                                                             # files 5-7, 7a-7c
STAGE=teachers bash training/train_teachers.sh                  # 3 families x 5 folds -> out-of-fold predictions
python -I labels/collect_oof.py oof_teacher_effnetv2s_progressive oof_teacher_effnetv2s_fixed oof_teacher_convnext_tiny
python -I labels/teacher_blend.py && python -I labels/teacher_blend.py --with-gold58          # files 8, 9
```

## 5. Train (train)

```bash
GPUS="0 1 2 3 4 5 6 7" bash training/train_final.sh      # the 8 final models -> MODEL_DIR (models/), one per GPU
ONLY=final1 GPUS="0 1 2 3" bash training/train_final.sh  # only the 4 models of final submission 1
```

Checkpoints go to `CHECKPOINT_DIR` (`runs/<name>/`), logs to `WORK_DIR/logs`, and each finished model is packaged
into `MODEL_DIR/<name>.pt`, the format `inference/` loads. Details per model: `training/README.md`.

## 6. Predict (predict)

```bash
python inference/infer.py                 # final submission 1 (4 models) -> SUBMISSION_DIR/submission.csv
ENSEMBLE=8 python inference/infer.py      # final submission 2 (8 models)
```

The test studies are read from `RAW_DATA_DIR` (`test.csv`, `test_series.csv`, `test_series/`), the models from
`MODEL_DIR`. To predict with the released weights instead of retrained ones, skip step 5:

```bash
kaggle datasets download songqizhou/rsna-knee-7th-place-models -p models --unzip
```

On Kaggle, the public notebooks `songqizhou/rsna-knee-7th-place-inference` (4 models) and
`songqizhou/rsna-knee-7th-place-inference-8-models` run the same code; `python inference/build_notebooks.py`
rebuilds them from `inference/*.py`.
