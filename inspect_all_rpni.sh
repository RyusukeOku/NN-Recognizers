#!/bin/bash

# Exit on any error
set -e

# Source the language definitions from the experiments directory
if [ ! -f experiments/include.bash ]; then
    echo "Error: experiments/include.bash not found." >&2
    exit 1
fi
source experiments/include.bash

# Define base paths
BASE_DATA_DIR="data/languages"
OUTPUT_DIR="results/rpni_automaton"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop through all defined languages
for lang in "${LANGUAGES[@]}"; do
    echo "----------------------------------------"
    echo "Processing language: $lang"
    echo "----------------------------------------"
    
    DATA_DIR="$BASE_DATA_DIR/$lang"
    VOCAB_PATH="$DATA_DIR/main.vocab"

    # Check if the data directory for the language exists
    if [ ! -d "$DATA_DIR" ]; then
        echo "Warning: Data directory not found for language '$lang' at '$DATA_DIR'. Skipping." >&2
        continue
    fi

    # Run the inspection and saving script
    python inspect_rpni_automaton.py \
        --data-dir "$DATA_DIR" \
        --vocab-path "$VOCAB_PATH" \
        --output-dir "$OUTPUT_DIR"
done

echo "----------------------------------------"
echo "All languages processed successfully."
