
import argparse
import sys
import torch
from pathlib import Path
from rayuela.fsa.state import State

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
    
    # Ensure all states are properly represented as State(int) and extract their int indices
    # This loop will filter out raw ints and fix recursive State objects if any
    cleaned_states_set = set()
    for s in fsa.Q:
        if isinstance(s, State):
            cleaned_states_set.add(int(s.idx))
        elif isinstance(s, int):
            cleaned_states_set.add(s)
        else:
            # Handle other unexpected types if necessary, or raise an error
            raise TypeError(f"Unexpected state type in fsa.Q: {type(s)}")
    states_int_idx = sorted(list(cleaned_states_set))

    initial_state_int_idx = None
    initial_state_tuple = next(fsa.I, None)
    if initial_state_tuple:
        q, w = initial_state_tuple
        if isinstance(q, State):
            initial_state_int_idx = int(q.idx)
        elif isinstance(q, int):
            initial_state_int_idx = q

    final_states_int_idx = []
    for q, w in fsa.F:
        if isinstance(q, State):
            final_states_int_idx.append(int(q.idx))
        elif isinstance(q, int):
            final_states_int_idx.append(q)

    arcs_data = []
    for p_obj in fsa.Q:
        p_idx = None
        if isinstance(p_obj, State):
            p_idx = int(p_obj.idx)
        elif isinstance(p_obj, int):
            p_idx = p_obj
        else:
            continue # Skip unexpected types

        for i_sym, q_obj, w_semiring in fsa.arcs(p_obj):
            q_idx = None
            if isinstance(q_obj, State):
                q_idx = int(q_obj.idx)
            elif isinstance(q_obj, int):
                q_idx = q_obj
            else:
                continue # Skip unexpected types

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
