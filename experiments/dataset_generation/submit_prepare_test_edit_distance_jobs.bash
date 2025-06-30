#!/bin/bash
set -euo pipefail
. experiments/include.bash

# ローカルGPUデバイスを指定 (例: cuda, cuda:0)
# prepare_test_edit_distance.bash がこのデバイス引数を
# 内部のPythonスクリプト等に渡すことを想定しています。
# GPUが不要な場合は "cpu" に変更してください。
LOCAL_DEVICE="gpu"

echo "Running test edit distance preparation locally on device: $LOCAL_DEVICE"

for language in repeat-01 dyck-2-3; do
  echo "Preparing test edit distance data for language: $language on $LOCAL_DEVICE"
  # submit_job呼び出しを直接実行に置き換え
  # poetry run は、prepare_test_edit_distance.bash 内のPythonスクリプトが
  # Poetry環境を必要とする場合に備えています。
  poetry run bash src/recognizers/string_sampling/prepare_test_edit_distance.bash \
    "$BASE_DIR" \
    "$language" \
    "$LOCAL_DEVICE"
  echo "Finished preparing test edit distance data for $language."
  echo "------------------------------------"
done

echo "All test edit distance preparations finished."
