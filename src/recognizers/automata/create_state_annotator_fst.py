import argparse
import sys
import torch
from pathlib import Path

# Add src directory to Python path to allow importing recognizers module
sys.path.append(str(Path(__file__).parent.parent.parent))

from recognizers.automata.finite_automaton import FiniteAutomatonContainer
from rayuela.fsa.fsa import FSA
from rayuela.fsa.fst import FST
from rayuela.base.semiring import Tropical

def convert_pt_to_fsa(automaton_container: FiniteAutomatonContainer, alphabet: list[str]) -> FSA:
    """
    Converts a FiniteAutomatonContainer object (from automaton.pt) into a Rayuela FSA object.
    """
    fsa = FSA()

    # 1. Add all states
    # We need to know the total number of states. Let's assume it's stored in num_states.
    for i in range(automaton_container.num_states):
        fsa.add_state(i)

    # 2. Set initial state
    fsa.set_initial(automaton_container.initial_state)

    # 3. Set final states
    for state in range(automaton_container.num_states):
        if automaton_container.is_accept_state(state):
            fsa.set_final(state)

    # 4. Add transitions
    # The alphabet list maps integer labels to string tokens.
    for transition in automaton_container.transitions():
        source_state = transition.source
        label_index = transition.label
        next_state = transition.target
        
        # Convert integer label to string token using the alphabet
        token = alphabet[label_index]
        
        # Rayuela FSA expects a weight, we'll use the default (one in Tropical semiring)
        fsa.add_arc(source_state, token, next_state, Tropical(0.0))

    return fsa

def create_state_annotator_fst_from_pt(pt_path: str, fst_path: str):
    """
    Reads an automaton.pt file, converts it to a Rayuela FSA, and then creates
    a state-annotating FST from it.
    """
    try:
        # Load the dictionary from the .pt file
        data = torch.load(pt_path, weights_only=False)
        automaton_container = data['automaton']
        alphabet = data['alphabet']
    except FileNotFoundError:
        print(f"Error: Automaton file not found at {pt_path}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, TypeError) as e:
        print(f"Error: The file {pt_path} does not have the expected format (dict with 'automaton' and 'alphabet'). Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert the loaded automaton to a Rayuela FSA
    print("Converting .pt automaton to Rayuela FSA...")
    fsa = convert_pt_to_fsa(automaton_container, alphabet)
    print("Conversion successful.")

    # Now, create the annotator FST from the Rayuela FSA
    annotator_fst = FST()

    for state in fsa.states:
        annotator_fst.add_state(state)

    annotator_fst.set_initial(fsa.initial_state)
    for final_state in fsa.final_states:
        annotator_fst.set_final(final_state)

    for arc in fsa.arcs:
        p, i, q, w = arc.source, arc.ilabel, arc.nextstate, arc.weight
        output_label = f"{i}_{p}"
        annotator_fst.add_arc(p, i, output_label, q, w)

    # Save the final FST
    annotator_fst.write(fst_path)
    print(f"State annotator FST successfully created and saved to {fst_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a state annotator FST from a project-specific automaton.pt file."
    )
    parser.add_argument(
        "automaton_pt_path",
        type=str,
        help="Path to the input automaton.pt file."
    )
    parser.add_argument(
        "output_fst_path",
        type=str,
        help="Path to save the output FST file."
    )
    args = parser.parse_args()

    create_state_annotator_fst_from_pt(args.automaton_pt_path, args.output_fst_path)