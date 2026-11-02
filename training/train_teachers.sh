#!/bin/bash
# The 5-fold models whose out-of-fold predictions the label pipeline uses (only needed to rebuild the labels from the
# LLM reads; the released label files already contain their effect). Fold model k trains on the other four folds and
# predicts its held-out fold k (and the 58 gold studies) with dense windows: CHECKPOINT_DIR/<run>/pred_denseT<T>.npz.
#   STAGE=synovitis  CA0-CA4: effnet_c1 recipe on the structured read (1_read_structured.csv), T 1.25; their
#                    out-of-fold Effusion sets the expected Synovitis rate of the per-hospital correction (labels/, file 4)
#   STAGE=teachers   the three teacher families on the sibling LLM labels (labels/, files 7a-7c), files 8 / 9:
#                    S4bL4b_f<k> effnet_fixed on 7c (T 1.75), B16L3b_f<k> effnet_progressive on 7b (T 1.75),
#                    CNXT_L1b_f<k> convnext (ConvNeXt-T) on 7a (T 1.25)
# usage (from the repository root): STAGE=synovitis|teachers [GPUS="0 1 2 3"] bash training/train_teachers.sh
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPUS=(${GPUS:-0})
others() { echo 0 1 2 3 4 | tr ' ' '\n' | grep -v "^$1$" | paste -sd,; }
JOBS=()
case "$STAGE" in
  synovitis) for k in 0 1 2 3 4; do JOBS+=("CA$k|effnet_c1|--labels 1_read_structured.csv --seed 42|1.25|$k"); done ;;
  teachers)
    for k in 0 1 2 3 4; do JOBS+=("B16L3b_f$k|effnet_progressive|--labels 7b_llm_label_variant.csv --seed 42|1.75|$k"); done
    for k in 0 1 2 3 4; do JOBS+=("S4bL4b_f$k|effnet_fixed|--labels 7c_llm_label_variant.csv --seed 42|1.75|$k"); done
    for k in 0 1 2 3 4; do JOBS+=("CNXT_L1b_f$k|convnext|--arch convnext_tiny.in12k_ft_in1k --labels 7a_llm_label_variant.csv --seed 43|1.25|$k"); done ;;
  *) echo "set STAGE=synovitis or STAGE=teachers"; exit 1 ;;
esac
LOGS=$($PY -c "from settings import path; print(path('WORK_DIR', 'logs'))"); mkdir -p "$LOGS"
declare -A BUSY
run() {   # name recipe "extra" T fold gpu
  if CUDA_VISIBLE_DEVICES=$6 $PY training/train.py --tag $1 --recipe $2 $3 --train_folds $(others $5) --val_fold $5 \
       > "$LOGS/train_$1.log" 2>&1 &&
     CUDA_VISIBLE_DEVICES=$6 $PY training/predict.py $1 --T $4 >> "$LOGS/train_$1.log" 2>&1; then
    echo "$(date '+%m-%d %H:%M') $1 done (GPU $6)"
  else
    echo "$(date '+%m-%d %H:%M') $1 FAILED (GPU $6), see $LOGS/train_$1.log"
  fi
}
for j in "${JOBS[@]}"; do
  IFS='|' read -r name recipe extra T k <<< "$j"
  while :; do   # first GPU without a running job
    for g in "${GPUS[@]}"; do
      p=${BUSY[$g]}
      if [ -z "$p" ] || ! kill -0 $p 2>/dev/null; then G=$g; break 2; fi
    done
    sleep 60
  done
  echo "$(date '+%m-%d %H:%M') start $name on GPU $G"
  run $name $recipe "$extra" $T $k $G &
  BUSY[$G]=$!
done
wait
echo "train_teachers ($STAGE) done"
