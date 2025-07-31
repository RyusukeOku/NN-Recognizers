set -euo pipefail

. src/recognizers/functions.bash

usage() {
  echo "Usage: $0 <base-directory> <language> <architecture> <loss-terms> \\
  <validation-data> <trial-no> [--no-progress]

Train and evaluate a neural network on a language.

  <base-directory>
    Directory under which all datasets and models are stored.
  <language>
    Name of the language to run on. Corresponds to the name of a directory
    under <base-directory>/languages/.
  <architecture>
    One of transformer, rnn, lstm.
  <loss-terms>
    Any of the following, joined by \`+\` characters:
    - rec: recognition (binary classification with binary cross-entropy loss)
    - lm: language modeling (cross-entropy loss of next symbol)
    - ns: next set prediction (binary cross-entropy loss of whether every
          symbol at every position is valid in the next position)
    Example: rec+lm for recognition and language modeling.
  <validation-data>
    Which validation set to use. One of: validation-short, validation-long.
  <trial-no>
    A number distinguishing this random restart.
  --no-progress
    Don't show progress messages.
"
}

random_sample() {
  python src/recognizers/neural_networks/random_sample.py "$@"
}

base_dir=${1-}
language=${2-}
architecture=${3-}
loss_terms=${4-}
validation_data=${5-}
trial_no=${6-}
if ! shift 6; then
  usage >&2
  exit 1
fi
progress_args=("$@")

language_dir=$(get_language_dir "$base_dir" "$language")

model_flags=($( \
  python src/recognizers/neural_networks/get_architecture_args.py \
    --architecture "$architecture" \
    --parameter-budget 64000 \
    --training-data "$language_dir" \
))

loss_term_flags=()
for loss_term in ${loss_terms//+/ }; do
  case $loss_term in
    rec) ;;
    lm)
      loss_term_flags+=( \
        --use-language-modeling-head \
        --language-modeling-loss-coefficient "$(random_sample --log 0.01 10)" \
      )
      ;;
    ns)
      loss_term_flags+=( \
        --use-next-symbols-head \
        --next-symbols-loss-coefficient "$(random_sample --log 0.01 10)" \
      )
      ;;
    *)
      echo "invalid loss term $loss_term" >&2
      exit 1
      ;;
  esac
done

model_dir=$(get_model_dir "$base_dir" "$language" "$architecture" "$loss_terms" "$validation_data" "$trial_no")

TRAINING_ARGS=(
  "--output" "$model_dir"
  "--training-data" "$language_dir"
  "--validation-data" "$validation_data"
  "--init-scale" "0.1"
  "${loss_term_flags[@]}"
  "--max-epochs" "1000"
  "--max-tokens-per-batch" "$(random_sample --int 128 4096)"
  "--optimizer" "Adam"
  "--initial-learning-rate" "$(random_sample --log 0.0001 0.01)"
  "--gradient-clipping-threshold" "5"
  "--early-stopping-patience" "10"
  "--learning-rate-patience" "5"
  "--learning-rate-decay-factor" "0.5"
  "--examples-per-checkpoint" "10000"
  "${progress_args[@]}"
)

case "$architecture" in
  transformer)
    EMBEDDING_SIZE=24 # 例: Transformerのトークン埋め込みサイズ
    STATE_EMBEDDING_SIZE=8 # 例: Transformerの状態埋め込みサイズ
    D_MODEL=$((EMBEDDING_SIZE + STATE_EMBEDDING_SIZE)) # d_modelは合計サイズ
    TRAINING_ARGS+=(
      "--architecture" "$architecture"
      "--num-layers" "$NUM_LAYERS"
      "--d-model" "$D_MODEL"
      "--num-heads" "$NUM_HEADS"
      "--feedforward-size" "$FEEDFORWARD_SIZE"
      "--dropout" "$DROPOUT"
      "--embedding-size" "$EMBEDDING_SIZE"
      "--state-embedding-size" "$STATE_EMBEDDING_SIZE"
    )
    ;;
  rnn)
    EMBEDDING_SIZE=60 # 例: RNNのトークン埋め込みサイズ
    STATE_EMBEDDING_SIZE=19 # 例: RNNの状態埋め込みサイズ
    HIDDEN_UNITS=$((EMBEDDING_SIZE + STATE_EMBEDDING_SIZE)) # hidden_unitsは合計サイズ
    TRAINING_ARGS+=(
      "--architecture" "$architecture"
      "--num-layers" "$NUM_LAYERS"
      "--hidden-units" "$HIDDEN_UNITS"
      "--dropout" "$DROPOUT"
      "--embedding-size" "$EMBEDDING_SIZE"
      "--state-embedding-size" "$STATE_EMBEDDING_SIZE"
    )
    ;;
  lstm)
    EMBEDDING_SIZE=30 # 例: LSTMのトークン埋め込みサイズ
    STATE_EMBEDDING_SIZE=10 # 例: LSTMの状態埋め込みサイズ
    HIDDEN_UNITS=$((EMBEDDING_SIZE + STATE_EMBEDDING_SIZE)) # hidden_unitsは合計サイズ
    TRAINING_ARGS+=(
      "--architecture" "$architecture"
      "--num-layers" "$NUM_LAYERS"
      "--hidden-units" "$HIDDEN_UNITS"
      "--dropout" "$DROPOUT"
      "--embedding-size" "$EMBEDDING_SIZE"
      "--state-embedding-size" "$STATE_EMBEDDING_SIZE"
    )
    ;;
  hybrid_csg)
    TRAINING_ARGS+=(
      "--architecture" "$architecture"
      "--hybrid-base-architecture" "$architecture" # ここは元のまま
      "--embedding-size" "$(case "$architecture" in transformer) echo 32;; rnn) echo 79;; lstm) echo 40;; esac)" # 元のembedding-size
      "--lba-hidden-size" "32"
      "--lba-n-steps" "150"
    )
    ;;
  *)
    echo "Unknown architecture: $architecture" >&2
    exit 1
    ;;
esac

poetry run python src/recognizers/neural_networks/train.py "${TRAINING_ARGS[@]}"
bash src/recognizers/neural_networks/evaluate.bash "$language_dir" "$model_dir"
