set -euo pipefail

. src/recognizers/functions.bash

usage() {
  echo "Usage: $0 <base-directory> <language> [--use-state-annotations <fst-path>]

Prepare the datasets for a language.

  <base-directory>
    Directory under which all datasets and models are stored.
  <language>
    Name of the language to prepare. Corresponds to the name of a directory
    under <base-directory>/languages/.
  --use-state-annotations <fst-path>
    Optional: Enable FST-based state annotations using the specified FST path.
"
}

base_dir=${1-}
language=${2-}

# Parse optional arguments
annotation_flags=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-state-annotations)
      annotation_flags+=(--use-state-annotations)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --use-state-annotations requires an argument." >&2
        usage >&2
        exit 1
      fi
      annotation_flags+=(--fst-annotator-path "$1")
      shift
      ;;
    *)
      break # End of optional arguments
      ;;
  esac
done

if ! shift 2; then
  usage >&2
  exit 1
fi

language_dir=$(get_language_dir "$base_dir" "$language")

# Optional datasets.
flags=()
for dataset in test-short-held-out test-edit-distance; do
  if [[ -e $language_dir/datasets/$dataset ]]; then
    flags+=(--more-data "$dataset")
  fi
done

python src/recognizers/neural_networks/prepare_data.py \
  --training-data "$language_dir" \
  --more-data validation-short \
  --more-data validation-long \
  --more-data test \
  "${flags[@]}" \
  "${annotation_flags[@]}" \
  --never-allow-unk \
  --use-next-symbols

