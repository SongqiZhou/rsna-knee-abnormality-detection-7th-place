# Labels

The test set has no reports, so the models can only be as good as the labels read out of the training reports. This
folder holds every step from the reports and images to the final training labels. Each step writes one file of the
public dataset [RSNA Knee LLM Labels](https://www.kaggle.com/datasets/songqizhou/rsna-knee-llm-labels): one row per
training study, the 12 labels as soft targets in [0, 1], no report text.

**Default starting point.** The LLM replies are not bit-reproducible, so training starts from the released files:

```bash
kaggle datasets download songqizhou/rsna-knee-llm-labels -p data/labels --unzip    # LABEL_DIR in SETTINGS.json
```

The final models train on `8_final_label.csv` (final submission 1) and `9_final_label_with_gold58_studies.csv`
(the twins of final submission 2). Everything below regenerates these files.

## The pipeline, in run order

| step | script | input | output | gold58 AUC |
|---|---|---|---|---|
| 1 | `read_structured.py` | reports + 24 key slices (API) | `1_read_structured.csv` (12 probabilities + 24 graded attributes) | .935 |
| 2 | `read_image_only.py` | every slice, no report (API) | `2_read_image_only.csv` | .919 |
| 3 | `read_verification.py` | every slice + the report as a draft that may contain errors (API) | `3_read_verification.csv` | .947 * |
| 4 | `synovitis_fix.py` | file 1 + out-of-fold Effusion of 5 fold models trained on file 1 | `4_read_structured_synovitis_fixed.csv` | .937 |
| 5 | `blend_reads.py` | files 4, 2 | `5_blend_reads_1_2.csv` (60 : 40, probability + rank) | .943 |
| 6 | `blend_reads.py` | files 5, 3 | `6_blend_reads_1_2_3.csv` (1 : 1 rank mean) | .949 |
| 7 | `blend_reads.py` | files 6, 5 | `7_llm_label.csv` = the LLM label: file 6 with Synovitis and Effusion from file 5 | .947 |
| 7a-c | `blend_reads.py` | files 5, 3 | `7a/7b/7c_llm_label_variant.csv`: sibling LLM labels (other read weights), one per teacher family | .946 / .947 / .946 |
| 8 | `teacher_blend.py` | file 7 + out-of-fold predictions of three 5-fold teacher families trained on 7a / 7b / 7c | `8_final_label.csv` = ½ LLM label + ½ teacher | .952 |
| 9 | `teacher_blend.py --with-gold58` | as 8, with the 58 gold studies added to the pool (their reads 1g-3g) | `9_final_label_with_gold58_studies.csv` | – |

\* The verification prompt was chosen on these 58 studies, so its score is optimistic. The gold58 scores compare each
file with the host labels of the 58 gold studies; the host labels were used only for these scores. No file is built
from them.

**Every blend keeps the prevalence of file 4.** Per label, the blend decides the order of the studies and the values
are the sorted values of file 4 (`common.remap`). Ranks are percentiles within the pool, with average ranks for ties.

**Step 4, Synovitis.** The structured read marks Synovitis almost exactly when the report uses the word, and how often
reports use it depends on the hospital. The scanner model stands in for the hospital. For every site with at least 30
studies, its Synovitis labels get one logit shift, solved so that the site's mean equals the rate expected from its
image Effusion level. The expected rate is the mean read-1 Synovitis label of the studies in the same quintile of
out-of-fold image Effusion. The order of studies within a site does not change.

**Steps 8-9, teacher.** Three 5-fold families: EfficientNetV2-S with the progressive window sampler (trained on 7b),
EfficientNetV2-S with the fixed sampler (7c), and ConvNeXt-T (7a). Each training study is scored only by the fold model
that did not see it. Per label, the three families' percentile ranks are averaged and that order is rank-mapped onto
the values of file 7. For file 9 the 58 gold studies join the pool: every rank and every marginal from step 5 on is
taken over the 4,407 studies. The gold rows get the mean of the five fold models, none of which trained on them.

## Commands

All scripts read and write through `SETTINGS.json` (`RAW_DATA_DIR`, `CACHE_DIR`, `LABEL_DIR`, `WORK_DIR`,
`CHECKPOINT_DIR`). Run from the repository root.

```bash
# steps 1-3: the three reads (API; each resumable, run them one after another)
export API_URL=<base URL of an OpenAI-compatible chat-completions endpoint>  API_KEY=<key>
python -I labels/read_structured.py               && python -I labels/read_structured.py --parse
python -I labels/read_image_only.py               && python -I labels/read_image_only.py --parse
python -I labels/read_verification.py             && python -I labels/read_verification.py --parse
# the 58 gold studies (needed for file 9 only)
python -I labels/read_structured.py --gold58      && python -I labels/read_structured.py --parse --gold58
python -I labels/read_image_only.py --gold58      && python -I labels/read_image_only.py --parse --gold58
python -I labels/read_verification.py --gold58    && python -I labels/read_verification.py --parse --gold58

# step 4: the training labels of the five image models (read 1's 12 probabilities) ...
python -I labels/read_structured.py --training-labels      # -> WORK_DIR/labels/read1_labels.csv
# ... train CA0-CA4 on them (training/README.md), then
python -I labels/collect_oof.py oof_image_model
python -I labels/synovitis_fix.py

# steps 5-7
python -I labels/blend_reads.py

# steps 8-9: train the three teacher families on 7a / 7b / 7c (training/README.md: CNXT_L1b, B16L3b, S4bL4b), then
python -I labels/collect_oof.py oof_teacher_effnetv2s_progressive oof_teacher_effnetv2s_fixed oof_teacher_convnext_tiny
python -I labels/teacher_blend.py
python -I labels/teacher_blend.py --with-gold58
```

Steps 4-9 take a few seconds on a CPU.

## The three reads

| read | system prompt | user message | images per study | tokens per study (input + output) | total, 4,349 studies |
|---|---|---|---|---|---|
| 1 structured | `prompts/system_structured.txt` | 3 montages + "Report:" + report | 9 sagittal + 9 coronal + 6 axial slices, 256 px tiles | ~4.7k | 20.4 M |
| 2 image-only | `prompts/system_image_only.txt` | every slice | ~92 frames of 384 px (58-116 for 90% of studies) | ~20k | 86.7 M |
| 3 verification | `prompts/system_verification.txt` | every slice + "Trainee's draft report (may contain errors; reference only):" + report | as read 2 | ~21k | 91.3 M |

- **Model.** `gpt-6-astra` through an OpenAI-compatible chat-completions endpoint, `max_tokens` 4000, default
  reasoning. Each study is one request and the reply is one JSON object: 24 integer grades (0-3) and 12 probabilities.
  The token totals do not include the prompt tests on the 58 gold studies.
- **Images.** All images come from the same 140 mm / 384 px slice cache the models train on (`preprocess/`), with
  right knees mirrored. Each plane uses its fat-suppressed series when the study has one. Each frame of reads 2-3
  costs about 190 input tokens, so the images are most of the cost. Series longer than 48 slices were evenly
  subsampled to 48 by the cache.
- **Ids.** Studies are sent under anonymous ids (`study_ids.csv`: `t*`/`v*` training studies, `g*` the gold studies),
  which also appear in the image titles. No StudyInstanceUID reaches the endpoint.
- **Replies.** Each reply is appended to `WORK_DIR/labels/<read>/rows.jsonl` as it returns, so a rerun resumes. Reads
  2-3 check the reply (integer grades, probabilities in [0, 1]) and retry otherwise. While the endpoint refuses (for
  example, the quota is used up), it is probed every 10 minutes. One worker by default.
- **Prompts.** All three state the host's label definitions and how the panel labelled. The verification prompt adds
  the trainee-draft framing and typical report errors per label, measured on the training set only (report-only labels
  against two image-based sources). Its Synovitis paragraph describes the per-hospital writing habit of step 4.
- **Gold58.** The 58 gold studies have the same three reads (`--gold58`). They are used only for file 9. Their
  structured read was sent without the words "(a missing plane is omitted)" in the user text.

## Checks

- **Requests.** For 10 studies (7 training, 3 gold), the scripts build the same requests as the runs that made the
  released files: the same system prompt, user text and PNG bytes. All 1,653 read-1 montages of 551 studies (151 with
  a short series) are byte-identical to the images that were sent. All 4,407 reports equal `train.csv`.
- **Parsing.** `--parse` on the saved replies reproduces files 1-3 byte for byte.
- **Steps 4-9.** From files 1-3, the gold58 reads and the out-of-fold predictions of the 20 fold models, the scripts
  reproduce files 4-9 (all ten) byte for byte.
- **Rendering.** The image titles are drawn with Pillow's default font, so the PNG bytes depend on the Pillow version.
  Byte identity was checked with Pillow 12.2.0.
