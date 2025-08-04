

import torch
import sys

try:
    # Load vocabulary
    vocab_data = torch.load("data/languages/even-pairs/main.vocab")
    tokens = vocab_data["tokens"]
    
    # Create integer-to-string mapping
    int_to_token = {i: token for i, token in enumerate(tokens)}

    # Load the prepared data
    prepared_data = torch.load("data/languages/even-pairs/main.prepared")

    print("--- Decoding main.prepared using main.vocab ---")
    if not prepared_data:
        print("main.prepared is empty.")
    else:
        # Decode and print the first 5 sentences
        for i, tensor in enumerate(prepared_data[:5]):
            decoded_tokens = [int_to_token.get(idx.item(), f"???_UNKNOWN_INDEX_{idx.item()}") for idx in tensor]
            print(f"Sentence {i+1}: {' '.join(decoded_tokens)}")

    print("\n--- Vocabulary (first 30 tokens) ---")
    if not tokens:
        print("Vocabulary is empty.")
    else:
        print(tokens[:30])

except FileNotFoundError as e:
    print(f"Error: Could not find a required file. {e}", file=sys.stderr)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)


