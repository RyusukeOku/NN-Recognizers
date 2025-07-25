import argparse
import sys
import torch
from pathlib import Path

# Add src directory to Python path to allow importing recognizers module
sys.path.append(str(Path(__file__).parent.parent.parent))

from recognizers.automata.finite_automaton import FiniteAutomatonContainer
from rayuela.fsa.fst import FST
from rayuela.base.semiring import Tropical

def create_state_annotator_fst_from_pt(pt_path: str, fst_path: str):
    """
    Reads an automaton.pt file, converts it to a Rayuela FSA using the
    built-in method, and saves the state-annotating FST data.
    """
    try:
        data = torch.load(pt_path, weights_only=False)
        automaton_container = data['automaton']
        alphabet = data['alphabet']
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
    initial_state = next(fsa.I, None)
    final_states = list(fsa.F)
    arcs = []
    for p in states:
        for i, q, w in fsa.arcs(p):
            output_label = f"{i}_{p}"
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
    args = parser.parse_args()
    create_state_annotator_fst_from_pt(args.automaton_pt_path, args.output_fst_path)