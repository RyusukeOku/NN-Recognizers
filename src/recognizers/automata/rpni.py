
import argparse
import pathlib
import sys
from collections import deque
from itertools import product

import torch
from rayuela.base.state import State
from rayuela.base.symbol import Sym
from rayuela.fsa.fsa import FSA


def load_data(language_dir: pathlib.Path) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """
    Loads positive and negative samples from the specified language directory.

    Args:
        language_dir: Path to the directory containing main.tok and labels.txt.

    Returns:
        A tuple containing:
        - A list of positive samples (each sample is a list of tokens).
        - A list of negative samples.
        - A sorted list of unique symbols (the alphabet).
    """
    strings_path = language_dir / 'main.tok'
    labels_path = language_dir / 'labels.txt'

    if not strings_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Data files not found in {language_dir}")

    with strings_path.open() as f_str, labels_path.open() as f_lbl:
        lines = f_str.readlines()
        labels = [int(l.strip()) for l in f_lbl.readlines()]

    positive_samples = []
    negative_samples = []
    alphabet = set()

    for string, label in zip(lines, labels):
        tokens = string.strip().split()
        if not tokens:
            continue
        
        alphabet.update(tokens)
        if label == 1:
            positive_samples.append(tokens)
        else:
            negative_samples.append(tokens)
            
    return positive_samples, negative_samples, sorted(list(alphabet))


def build_pta(positive_samples: list[list[str]]) -> FSA:
    """
    Builds a Prefix Tree Acceptor (PTA) from a set of positive samples.

    Args:
        positive_samples: A list of positive samples.

    Returns:
        An FSA object representing the PTA.
    """
    fsa = FSA()
    state_counter = 0
    start_state = State(state_counter)
    fsa.add_state(start_state)
    state_counter += 1
    fsa.set_I(start_state)

    for sample in positive_samples:
        current_state = start_state
        for token in sample:
            symbol = Sym(token)
            next_state = None
            for s, q, _ in fsa.arcs(current_state):
                if s == symbol:
                    next_state = q
                    break
            
            if next_state is None:
                next_state = State(state_counter)
                fsa.add_state(next_state)
                state_counter += 1
                fsa.add_arc(current_state, symbol, next_state, fsa.R.one)
            
            current_state = next_state
        # Add final state with the semiring's multiplicative identity (weight 1)
        fsa.add_F(current_state, fsa.R.one)
        
    return fsa


def accepts(fsa: FSA, sample: list[str]) -> bool:
    """
    Checks if the given FSA accepts a sample string.
    Handles non-determinism by tracking a set of current states.
    """
    current_states = set(fsa.I)
    for token in sample:
        symbol = Sym(token)
        next_states = set()
        for state in current_states:
            for s, q, _ in fsa.arcs(state):
                if s == symbol:
                    next_states.add(q)
        current_states = next_states
        if not current_states:
            return False
    
    # A string is accepted if any of the current states are in the set of final states.
    return not current_states.isdisjoint(fsa.F.keys())


def merge_states(fsa: FSA, q_from: State, q_to: State) -> FSA:
    """
    Merges state q_from into q_to in a new FSA.
    """
    new_fsa = FSA()
    state_map = {}

    # Create new states, mapping q_from to q_to
    for q in fsa.Q:
        if q != q_from:
            # Create a new State object with the same index from the old FSA
            new_q = State(q.idx)
            new_fsa.add_state(new_q)
            state_map[q] = new_q
    
    # The new state corresponding to the merge target q_to
    q_to_new = state_map[q_to]

    # Remap arcs
    for p in fsa.Q:
        for symbol, r, weight in fsa.arcs(p):
            # Map source and destination states to the new FSA's states
            p_new = q_to_new if p == q_from else state_map[p]
            r_new = q_to_new if r == q_from else state_map[r]
            
            # Avoid adding duplicate arcs that might result from the merge
            is_duplicate = False
            for s_existing, r_existing, _ in new_fsa.arcs(p_new):
                if s_existing == symbol and r_existing == r_new:
                    is_duplicate = True
                    break
            if not is_duplicate:
                new_fsa.add_arc(p_new, symbol, r_new, weight)

    # Remap initial and final states
    for i_state in fsa.I:
        i_new = q_to_new if i_state == q_from else state_map[i_state]
        new_fsa.set_I(i_new)

    # Iterate over final states and their weights
    for f_state, f_weight in fsa.F.items():
        f_new = q_to_new if f_state == q_from else state_map[f_state]
        # If the new final state is not already final, add it with the original weight.
        if f_new not in new_fsa.F:
            new_fsa.add_F(f_new, f_weight)
            
    return new_fsa


def rpni(positive_samples: list[list[str]], negative_samples: list[list[str]]) -> FSA:
    """
    Performs the RPNI algorithm to infer an FSA.
    """
    fsa = build_pta(positive_samples)
    
    # States need to be ordered for the merging loop
    ordered_states = sorted(list(fsa.Q), key=lambda s: s.idx)
    
    while True:
        merged_in_iteration = False
        for i in range(len(ordered_states)):
            for j in range(i):
                q_i = ordered_states[i]
                q_j = ordered_states[j]

                # Try merging q_i into q_j
                merged_fsa = merge_states(fsa, q_i, q_j)
                
                is_consistent = True
                for neg_sample in negative_samples:
                    if accepts(merged_fsa, neg_sample):
                        is_consistent = False
                        break
                
                if is_consistent:
                    fsa = merged_fsa
                    ordered_states = sorted(list(fsa.Q), key=lambda s: s.idx)
                    merged_in_iteration = True
                    break  # Restart the loop with the new FSA
            if merged_in_iteration:
                break
        
        if not merged_in_iteration:
            break # No more valid merges found

    return fsa


def to_dict_container(fsa: FSA, alphabet: list[str]) -> dict:
    """
    Converts a Rayuela FSA to a dictionary format compatible with FSA_integrated_input_layer.
    """
    state_map = {state: i for i, state in enumerate(sorted(list(fsa.Q), key=lambda s: s.idx))}
    symbol_map = {symbol: i for i, symbol in enumerate(alphabet)}

    num_states = len(state_map)
    initial_state = state_map[next(iter(fsa.I))]
    final_states = [state_map[s] for s in fsa.F.keys()]
    
    transitions = []
    for p in fsa.Q:
        for symbol, q, _ in fsa.arcs(p):
            p_id = state_map[p]
            q_id = state_map[q]
            symbol_id = symbol_map[symbol.val]
            transitions.append((p_id, symbol_id, q_id))

    return {
        'num_states': num_states,
        'initial_state': initial_state,
        'final_states': final_states,
        'transitions': transitions,
        'alphabet': alphabet
    }


def main():
    parser = argparse.ArgumentParser(description='Infer an FSA from examples using RPNI.')
    parser.add_argument('--language-dir', type=pathlib.Path, required=True,
                        help='Directory with language data (main.tok, labels.txt).')
    parser.add_argument('--output-path', type=pathlib.Path, required=True,
                        help='Path to save the inferred FSA data.')
    args = parser.parse_args()

    print(f"Loading data from: {args.language_dir}", file=sys.stderr)
    pos_samples, neg_samples, alphabet = load_data(args.language_dir)
    print(f"Found {len(pos_samples)} positive, {len(neg_samples)} negative samples.", file=sys.stderr)
    print(f"Alphabet size: {len(alphabet)}", file=sys.stderr)

    print("Running RPNI algorithm...", file=sys.stderr)
    inferred_fsa = rpni(pos_samples, neg_samples)
    print("RPNI finished.", file=sys.stderr)

    print("Converting FSA to container format...", file=sys.stderr)
    fsa_container = to_dict_container(inferred_fsa, alphabet)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving inferred FSA to: {args.output_path}", file=sys.stderr)
    torch.save(fsa_container, args.output_path)
    print("Done.", file=sys.stderr)


if __name__ == '__main__':
    main()
