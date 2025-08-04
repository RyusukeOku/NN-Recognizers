import argparse
import sys
import torch
from pathlib import Path

# Add src directory to Python path to allow importing recognizers module
sys.path.append(str(Path(__file__).parent.parent.parent))

from recognizers.automata.finite_automaton import FiniteAutomatonContainer
from rayuela.fsa.fst import FST
from rayuela.base.semiring import Tropical

def create_state_annotator_fst_from_pt(pt_path: str, fst_path: str, explicit_alphabet: list[str] | None = None):
    """
    Reads an automaton.pt file, converts it to a Rayuela FSA using the
    built-in method, and saves the state-annotating FST data.
    """
    try:
        data = torch.load(pt_path, weights_only=False)
        automaton_container = data['automaton']
        
        # Determine alphabet
        if 'alphabet' in data:
            alphabet = data['alphabet']
            print("Alphabet found in .pt file.")
        elif explicit_alphabet:
            alphabet = explicit_alphabet
            print(f"Using explicit alphabet: {alphabet}")
        else:
            raise ValueError("Alphabet not found in .pt file and not provided via --alphabet argument.")

    except FileNotFoundError:
        print(f"Error: Automaton file not found at {pt_path}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, TypeError) as e:
        print(f"Error: The file {pt_path} does not have the expected format. Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert to Rayuela FSA using the object's own method
    print("Converting .pt automaton to Rayuela FSA...")
    fsa = automaton_container.to_rayuela_fsa(alphabet)
    print("Conversion successful.")

    # Build the serializable data dictionary for the annotator FST
    print("Extracting data to build FST...")
    states = list(fsa.Q)
    
    # Extract only the state from the (state, weight) tuple
    initial_state_tuple = next(fsa.I, None)
    initial_state = initial_state_tuple[0] if initial_state_tuple else None

    # Extract only the states from the (state, weight) tuples
    final_states = [s for s, w in fsa.F.items()]

    arcs = []
    for p in states:
        for i, q, w in fsa.arcs(p):
            output_label = f"{i}_{p}"
            # Check if the next state (q) is a final state
            if q in final_states:
                output_label = f"{output_label}_F"
            arcs.append((p, i, output_label, q, w.value))

    fst_data = {
        'states': states,
        'initial_state': initial_state,
        'final_states': final_states,
        'arcs': arcs,
        'semiring_type': 'tropical'
    }

    torch.save(fst_data, fst_path)
    print(f"State annotator FST data successfully created and saved to {fst_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a state annotator FST from a project-specific automaton.pt file."
    )
    parser.add_argument("automaton_pt_path", type=str, help="Path to the input automaton.pt file.")
    parser.add_argument("output_fst_path", type=str, help="Path to save the output FST data file.")
    parser.add_argument(
        "--alphabet",
        type=str,
        help="Comma-separated string of alphabet symbols (e.g., '0,1,a,b'). Required if alphabet is not in .pt file."
    )
    args = parser.parse_args()

    explicit_alphabet_list = None
    if args.alphabet:
        explicit_alphabet_list = args.alphabet.split(',')

    create_state_annotator_fst_from_pt(args.automaton_pt_path, args.output_fst_path, explicit_alphabet_list)