#!/bin/bash
# The final models, trained on the full training set, then packaged for inference/ (MODEL_DIR/<name>.pt).
#   final submission 1 (4 models): labels 8_final_label.csv, folds 0-4 (4,349 studies)
#   final submission 2 (8 models): the same 4 + a twin of each, labels 9_final_label_with_gold58_studies.csv, folds -1..4
#                                  (4,407 studies: the training set + the 58 gold studies with LLM labels), another seed
# usage (from the repository root): [GPUS="0 1 2 3"] [ONLY=final1] bash training/train_final.sh
# One GPU per model; with several GPUs in GPUS the models train in parallel, one per GPU.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPUS=(${GPUS:-0})
MON=training/monitor_labels_fold0_first100.csv
F1=8_final_label.csv
F2=9_final_label_with_gold58_studies.csv
# name | recipe | extra arguments | inference attention temperature
JOBS=(
  "effnetv2s_fixed|effnet_fixed|--labels $F1 --seed 3407 --val_limit 100 --ref_labels $MON --ref_overrides 1|1.75"
  "effnetv2s_progressive|effnet_progressive|--labels $F1 --seed 3407 --val_limit 100 --ref_labels $MON --ref_overrides 1|1.75"
  "convnext_small|convnext|--arch convnext_small.in12k_ft_in1k --labels $F1 --seed 114514|1.25"
  "convnext_tiny|convnext|--arch convnext_tiny.in12k_ft_in1k --labels $F1 --seed 3407|1.25"
)
if [ "$ONLY" != "final1" ]; then JOBS+=(
  "effnetv2s_fixed_twin|effnet_fixed|--labels $F2 --train_folds=-1,0,1,2,3,4 --seed 114514 --val_limit 100|1.75"
  "effnetv2s_progressive_twin|effnet_progressive|--labels $F2 --train_folds=-1,0,1,2,3,4 --seed 114514 --val_limit 100|1.75"
  "convnext_small_twin|convnext|--arch convnext_small.in12k_ft_in1k --labels $F2 --train_folds=-1,0,1,2,3,4 --seed 3407|1.25"
  "convnext_tiny_twin|convnext|--arch convnext_tiny.in12k_ft_in1k --labels $F2 --train_folds=-1,0,1,2,3,4 --seed 114514|1.25"
); fi
LOGS=$($PY -c "from settings import path; print(path('WORK_DIR', 'logs'))"); mkdir -p "$LOGS"
declare -A BUSY
run() {   # name recipe "extra" T gpu
  if CUDA_VISIBLE_DEVICES=$5 $PY training/train.py --tag $1 --recipe $2 $3 > "$LOGS/train_$1.log" 2>&1 &&
     $PY training/package.py $1 --att_temp $4 >> "$LOGS/train_$1.log" 2>&1; then
    echo "$(date '+%m-%d %H:%M') $1 done (GPU $5)"
  else
    echo "$(date '+%m-%d %H:%M') $1 FAILED (GPU $5), see $LOGS/train_$1.log"
  fi
}
for j in "${JOBS[@]}"; do
  IFS='|' read -r name recipe extra T <<< "$j"
  while :; do   # first GPU without a running job
    for g in "${GPUS[@]}"; do
      p=${BUSY[$g]}
      if [ -z "$p" ] || ! kill -0 $p 2>/dev/null; then G=$g; break 2; fi
    done
    sleep 60
  done
  echo "$(date '+%m-%d %H:%M') start $name on GPU $G"
  run $name $recipe "$extra" $T $G &
  BUSY[$G]=$!
done
wait
echo "train_final done -> $($PY -c "from settings import path; print(path('MODEL_DIR'))")"
