#!/bin/bash
set -euo pipefail
. experiments/include.bash

# このスクリプトで呼び出される prepare_hand_coded_dataset.bash は
# 主にCPUで動作するデータ準備処理です。
# そのため、デバイス指定は必須ではありませんが、他のスクリプトとの
# 一貫性のために変数を残しています。
# LOCAL_DEVICE="cpu" # 通常はこちらで問題ありません。
LOCAL_DEVICE="cuda"  # もし将来的にGPUを利用する処理が含まれる場合に備える場合

echo "Running hand-coded dataset preparation locally"

for language in "${HAND_WRITTEN_LANGUAGES[@]}"; do
  echo "Preparing dataset for hand-coded language: $language"
  # submit_job呼び出しを直接実行に置き換え
  # prepare_hand_coded_dataset.bash は元々デバイス引数を取らないため、
  # LOCAL_DEVICE 変数はこのスクリプト内では直接使用されません。
  # もし prepare_hand_coded_dataset.bash がデバイス引数を受け付けるように
  # 変更された場合は、適切な引数を渡すようにしてください。
  poetry run bash src/recognizers/string_sampling/prepare_hand_coded_dataset.bash \
    "$BASE_DIR" \
    "$language"
  echo "Finished preparing $language."
  echo "------------------------------------"
done

echo "All hand-coded dataset preparations finished."s