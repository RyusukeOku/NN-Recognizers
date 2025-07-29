#!/bin/bash
set -euo pipefail
. experiments/include.bash

# ローカルGPUデバイスを指定 (例: cuda, cuda:0)
LOCAL_DEVICE="gpu"

echo "Running dataset preparation locally on device: $LOCAL_DEVICE"

for language in "${FINITE_AUTOMATON_LANGUAGES[@]}"; do
  echo "Preparing dataset for language: $language on $LOCAL_DEVICE"
  # submit_job呼び出しを直接実行に置き換え
  # poetry run は、prepare_automaton_dataset.bash内のPythonスクリプトが
  # Poetry環境を必要とする場合に備えていますが、
  # prepare_automaton_dataset.bash が内部でpoetry run python ... を
  # 適切に呼び出していれば、単なる bash ... でも構いません。
  # ここでは、より安全なpoetry run bash ... を採用します。
  poetry run bash src/recognizers/string_sampling/prepare_automaton_dataset.bash \
    "$BASE_DIR" \
    "$language" \
    "$LOCAL_DEVICE" \
    --use-state-annotations "$language_dir/annotator.fst"
  echo "Finished preparing $language."
  echo "------------------------------------"
done

echo "All dataset preparations finished."