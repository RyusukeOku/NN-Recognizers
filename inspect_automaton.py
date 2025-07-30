import torch
import sys
def inspect_automaton_file(file_path):
    """
    Reads an automaton file and prints its content.
    """
    try:
        data = torch.load(file_path, weight_only=False)
        print("Successfully loaded the automaton object.")
        print("Type:", type(data))
         # Assuming 'data' is the FiniteAutomatonContainer, let's inspect its attributes
        if hasattr(data, '__dict__'):
            print("\nAttributes of the object:")
            for key, value in data.__dict__.items():
                print(f"  - {key}: (type: {type(value)})")
        # If it's a dictionary, let's see the keys
        if isinstance(data, dict):
            print("\nContents of the dictionary:")
            for key, value in data.items():
                print(f"  - Key: '{key}', Type: {type(value)}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_automaton_file.py <path_to_automaton.pt>")
        sys.exit(1)
    inspect_automaton_file(sys.argv[1])