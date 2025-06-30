#!/bin/bash
set -euo pipefail
. experiments/include.bash
. src/recognizers/functions.bash # get_model_dir, get_language_dir を利用

# ローカルGPUデバイスを指定 (例: cuda, cuda:0)
LOCAL_DEVICE="cuda"
# Pythonスクリプトに渡すためのデバイスフラグ
DEVICE_FLAG="--device $LOCAL_DEVICE"

echo "Running re-evaluation locally on device: $LOCAL_DEVICE"

for language in "${LANGUAGES[@]}"; do
  language_dir=$(get_language_dir "$BASE_DIR" "$language")
  for architecture in "${ARCHITECTURES[@]}"; do
    for loss_terms in "${LOSS_TERMS[@]}"; do
      for validation_data in "${VALIDATION_SETS[@]}"; do
        for trial_no in "${TRIALS[@]}"; do
          model_dir=$(get_model_dir "$BASE_DIR" "$language" "$architecture" "$loss_terms" "$validation_data" "$trial_no")
          JOB_NAME="evaluate+${language}+${architecture}+${loss_terms//+/_}+${validation_data}+${trial_no}"
          echo "Starting job: $JOB_NAME"
          
          if [ ! -d "$model_dir" ]; then
            echo "Model directory $model_dir not found. Skipping."
            continue
          fi

          # submit_job呼び出しを直接実行に置き換え
          # evaluate.bashがDEVICE_FLAGを内部のevaluate.pyに
          # --device オプションとして渡すことを想定しています。
          poetry run bash src/recognizers/neural_networks/evaluate.bash \
            "$language_dir" \
            "$model_dir" \
            $DEVICE_FLAG
            
          echo "Finished job: $JOB_NAME"
          echo "------------------------------------"
        done
      done
    done
  done
done

echo "All re-evaluation jobs finished."