#!/bin/bash
set -euo pipefail
. experiments/include.bash

# ローカルGPUデバイスを指定 (例: cuda, cuda:0)
LOCAL_DEVICE="cuda"
# Pythonスクリプトに渡すためのデバイスフラグ
DEVICE_FLAG="--device $LOCAL_DEVICE"

echo "Running training and evaluation locally on device: $LOCAL_DEVICE"

for language in "${LANGUAGES[@]}"; do
  for architecture in "${ARCHITECTURES[@]}"; do
    for loss_terms in "${LOSS_TERMS[@]}"; do
      for validation_data in "${VALIDATION_SETS[@]}"; do
        for trial_no in "${TRIALS[@]}"; do
          JOB_NAME="train+${language}+${architecture}+${loss_terms//+/_}+${validation_data}+${trial_no}"
          echo "Starting job: $JOB_NAME"
          
          # submit_job呼び出しを直接実行に置き換え
          # train_and_evaluate.bashがDEVICE_FLAGを内部のtrain.pyおよびevaluate.pyに
          # --device オプションとして渡すことを想定しています。
          # 例: poetry run python src/recognizers/neural_networks/train.py ... --device cuda
          # もしtrain_and_evaluate.bashが追加の引数をそのままpythonスクリプトに渡さない場合、
          # train_and_evaluate.bash自体を修正する必要があります。
          poetry run bash src/recognizers/neural_networks/train_and_evaluate.bash \
            "$BASE_DIR" \
            "$language" \
            "$architecture" \
            "$loss_terms" \
            "$validation_data" \
            "$trial_no" \
            $DEVICE_FLAG \
            --no-progress # --no-progress は元のスクリプトのオプション
            
          echo "Finished job: $JOB_NAME"
          echo "------------------------------------"
        done
      done
    done
  done
done

echo "All training and evaluation jobs finished."