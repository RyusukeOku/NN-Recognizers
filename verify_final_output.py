import torch
import sys

print("--- Final Verification of Annotation Output ---")
try:
    # Load the new vocabulary
    vocab_data = torch.load("data/languages/even-pairs/main.vocab")
    tokens = vocab_data["tokens"]
    int_to_token = {i: token for i, token in enumerate(tokens)}

    # Load the new prepared data
    prepared_data = torch.load("data/languages/even-pairs/main.prepared")

    print("\n--- Decoding main.prepared (first 5 sentences) ---")
    if not prepared_data:
        print("main.prepared is empty.")
    else:
        for i, sentence_indices in enumerate(prepared_data[:5]):
            # Data is now a list of ints, no .item() needed
            decoded_tokens = [int_to_token.get(idx, f"???_IDX_{idx}") for idx in sentence_indices]
            print(f"Sentence {i+1}: {' '.join(decoded_tokens)}")

    print("\n--- Vocabulary (first 50 tokens) ---")
    if not tokens:
        print("Vocabulary is empty.")
    else:
        print("Vocabulary size:", len(tokens))
        print(tokens[:50])

except FileNotFoundError as e:
    print(f"Error: Could not find a required file. {e}", file=sys.stderr)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
