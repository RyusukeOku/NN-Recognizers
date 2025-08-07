import argparse
import sys
import torch
from pathlib import Path
from rayuela.base.state import State

# Add src directory to Python path to allow importing recognizers module
sys.path.append(str(Path(__file__).parent.parent.parent))

from recognizers.automata.finite_automaton import FiniteAutomatonContainer
from rayuela.fsa.fst import FST
from rayuela.base.semiring import Tropical

def get_state_index(state_obj):
    """Robustly extracts an integer index from various state representations."""
    if isinstance(state_obj, int):
        return state_obj
    if isinstance(state_obj, State):
        # Recursively handle nested State objects or other types in idx
        return get_state_index(state_obj.idx)
    if isinstance(state_obj, tuple) and len(state_obj) > 0:
        # Assumes the index is the first element of the tuple, e.g., (index, weight)
        return get_state_index(state_obj[0])
    try:
        # Final attempt to cast to int
        return int(state_obj)
    except (ValueError, TypeError):
        raise TypeError(f"Could not extract integer index from state: {state_obj} (type: {type(state_obj)})")

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
    
    # Use the helper function to robustly get integer indices for all states
    states_int_idx = sorted(list({get_state_index(s) for s in fsa.Q}))

    initial_state_int_idx = None
    initial_state_tuple = next(fsa.I, None)
    if initial_state_tuple:
        q, w = initial_state_tuple
        initial_state_int_idx = get_state_index(q)

    final_states_int_idx = [get_state_index(q) for q, w in fsa.F]

    arcs_data = []
    for p_obj in fsa.Q:
        try:
            p_idx = get_state_index(p_obj)
        except TypeError:
            continue # Skip states we can't index

        for i_sym, q_obj, w_semiring in fsa.arcs(p_obj):
            try:
                q_idx = get_state_index(q_obj)
            except TypeError:
                continue # Skip arcs to states we can't index

            output_label = f"{i_sym}_{p_idx}"
            # Check if the next state (q_idx) is a final state
            if q_idx in final_states_int_idx:
                output_label = f"{output_label}_F"
            arcs_data.append((p_idx, str(i_sym), output_label, q_idx, w_semiring.value))

    fst_data = {
        'states': states_int_idx,
        'initial_state': initial_state_int_idx,
        'final_states': final_states_int_idx,
        'arcs': arcs_data,
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