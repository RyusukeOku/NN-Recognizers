set -euo pipefail

. src/recognizers/functions.bash

usage() {
  echo "Usage: $0 <base-directory> <language> <architecture> <loss-terms> \\
  <validation-data> <trial-no> [--no-progress] [--fsa_state_integration]"

  echo "Train and evaluate a neural network on a language."

  echo "  <base-directory>"
  echo "    Directory under which all datasets and models are stored."
  echo "  <language>"
  echo "    Name of the language to run on. Corresponds to the name of a directory"
  echo "    under <base-directory>/languages/."
  echo "  <architecture>"
  echo "    One of transformer, rnn, lstm."
  echo "  <loss-terms>"
  echo "    Any of the following, joined by 
+ characters:"
  echo "    - rec: recognition (binary classification with binary cross-entropy loss)"
  echo "    - lm: language modeling (cross-entropy loss of next symbol)"
  echo "    - ns: next set prediction (binary cross-entropy loss of whether every"
  echo "          symbol at every position is valid in the next position)"
  echo "    Example: rec+lm for recognition and language modeling."
  echo "  <validation-data>"
  echo "    Which validation set to use. One of: validation-short, validation-long."
  echo "  <trial-no>"
  echo "    A number distinguishing this random restart."
  echo "  --no-progress"
  echo "    Don't show progress messages."
  echo "  --fsa_state_integration"
  echo "    Use FSA state integration."
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

fsa_flags=()
progress_args=()
for arg in "$@"; do
  case "$arg" in
    --fsa_state_integration)
      fsa_flags+=(--fsa_state_integration)
      ;;
    *)
      progress_args+=("$arg")
      ;;
  esac
done

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
    rec) ;; \
    lm) 
      loss_term_flags+=( \
        --use-language-modeling-head \
        --language-modeling-loss-coefficient "$(random_sample --log 0.01 10)" \
      )
      ;; \
    ns) 
      loss_term_flags+=( \
        --use-next-symbols-head \
        --next-symbols-loss-coefficient "$(random_sample --log 0.01 10)" \
      )
      ;; \
    *)
      echo "invalid loss term $loss_term" >&2
      exit 1
      ;; 
  esac
done

model_dir=$(get_model_dir "$base_dir" "$language" "$architecture" "$loss_terms" "$validation_data" "$trial_no")
python src/recognizers/neural_networks/train.py \
  --output "$model_dir" \
  --training-data "$language_dir" \
  --validation-data "$validation_data" \
  --language "$language" \
  --architecture "$architecture" \
  "${fsa_flags[@]}" \
  "${model_flags[@]}" \
  --init-scale 0.1 \
  "${loss_term_flags[@]}" \
  --max-epochs 1000 \
  --max-tokens-per-batch "$(random_sample --int 128 4096)" \
  --optimizer Adam \
  --initial-learning-rate "$(random_sample --log 0.0001 0.01)" \
  --gradient-clipping-threshold 5 \
  --early-stopping-patience 10 \
  --learning-rate-patience 5 \
  --learning-rate-decay-factor 0.5 \
  --examples-per-checkpoint 10000 \
  "${progress_args[@]}"
bash src/recognizers/neural_networks/evaluate.bash "$language_dir" "$model_dir" "${fsa_flags[@]}"
