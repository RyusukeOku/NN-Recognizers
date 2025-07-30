
import argparse
import json
import pathlib
import sys

import torch

# Imports for FST annotation
from rayuela.fsa.fsa import FSA
from rayuela.fsa.fst import FST
from rayuela.base.semiring import Tropical, Real

from rau.tasks.common.data_preparation import (
    add_prepare_data_args,
    get_token_types_in_file,
    prepare_file,
    validate_prepare_data_args,
    get_token_types
)
from rau.tasks.language_modeling.vocabulary import build_softmax_vocab
from rau.vocab import ToIntVocabularyBuilder, ToStringVocabularyBuilder


# --- Functions for FST State Annotation ---

def reconstruct_fst_from_data(fst_data: dict) -> FST:
    """Reconstructs a Rayuela FST object from a saved data dictionary."""
    semiring_map = {
        'tropical': Tropical,
        'real': Real
    }
    semiring = semiring_map.get(fst_data['semiring_type'])
    if semiring is None:
        raise ValueError(f"Unsupported semiring type: {fst_data['semiring_type']}")

    fst = FST(R=semiring)
    for state in fst_data['states']:
        fst.add_state(state)
    
    if fst_data['initial_state'] is not None:
        fst.set_I(fst_data['initial_state'])
    
    for final_state in fst_data['final_states']:
        fst.add_F(final_state, semiring(0.0))
    
    for p, i, o, q, w_val in fst_data['arcs']:
        fst.add_arc(p, i, o, q, semiring(w_val))
        
    return fst

def annotate_string_and_get_states(tokens: list[str], annotator_fst: FST) -> tuple[list[str], list[int]]:
    """Annotates a list of tokens using the provided FST and returns state IDs."""
    if not tokens:
        return [], []

    # Manually construct a linear FSA from the input tokens
    input_fsa = FSA(R=annotator_fst.R)
    num_states = len(tokens) + 1
    for i in range(num_states):
        input_fsa.add_state(i)
    input_fsa.set_I(0)
    input_fsa.add_F(num_states - 1, annotator_fst.R(0.0))

    for i, token in enumerate(tokens):
        input_fsa.add_arc(i, token, i + 1, annotator_fst.R(0.0))

    try:
        composed_fst = annotator_fst.compose(input_fsa)
        best_path = composed_fst.shortest_path()
        if not best_path or not best_path.I:
            # Fallback if no path is found
            return tokens, [annotator_fst.I] * len(tokens)

        # Traverse the best path to extract annotated tokens and state IDs
        annotated_tokens = []
        state_ids = []
        
        # The states in the composed FST are tuples: (state_from_annotator, state_from_input)
        # We start from the initial state of the best path.
        current_state = best_path.I
        
        # Create a map from source state to arc for easy traversal
        # Since it's a single path, each state (except the final one) has one outgoing arc.
        arc_map = {arc.source: arc for arc in best_path.arcs}

        while current_state in arc_map:
            arc = arc_map[current_state]
            annotated_tokens.append(arc.olabel)
            # The destination state of the arc is a tuple, we want the first element.
            state_ids.append(arc.dest[0])
            current_state = arc.dest
        
        # Ensure the lengths match, otherwise something is wrong.
        if len(state_ids) != len(tokens):
             return tokens, [annotator_fst.I] * len(tokens)

        return annotated_tokens, state_ids

    except Exception:
        # Fallback in case of any error during composition or pathfinding
        return tokens, [annotator_fst.I] * len(tokens)


def get_annotated_token_types_in_file(path, unk_string, annotator_fst):
    """Reads tokens, annotates them, and returns the set of unique annotated tokens."""
    def generate_annotated_tokens():
        with path.open() as fin:
            for line in fin:
                tokens = line.strip().split()
                annotated_tokens, _ = annotate_string_and_get_states(tokens, annotator_fst)
                for token in annotated_tokens:
                    yield token
    return get_token_types(generate_annotated_tokens(), unk_string)

def prepare_annotated_file_and_states(vocab, annotator_fst, strings_pair, states_pair):
    """Annotates strings in a file and saves them and their state IDs as integerized tensors."""
    input_path, output_path = strings_pair
    states_output_path = states_pair[1]
    
    print(f'preparing annotated tokens in {input_path} => {output_path}', file=sys.stderr)
    print(f'preparing states in {input_path} => {states_output_path}', file=sys.stderr)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    states_output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as fin:
        tokens_data = []
        states_data = []
        for line in fin:
            tokens = line.strip().split()
            annotated_tokens, state_ids_gen = annotate_string_and_get_states(tokens, annotator_fst)
            try:
                tokens_data.append(torch.tensor([vocab.to_int(t) for t in annotated_tokens]))
                # state_ids_gen might be a list of generators, extract the integer from each.
                # Also, ensure all states are integers before creating the tensor.
                materialized_state_ids = [int(s) for s in state_ids_gen]
                states_data.append(torch.tensor(materialized_state_ids, dtype=torch.long))
            except (KeyError, TypeError) as e:
                raise ValueError(f'{input_path}: error processing line: {line.strip()}\nError: {e}')
        torch.save(tokens_data, output_path)
        torch.save(states_data, states_output_path)

# --- Original Functions (from project) ---

def get_token_types_in_next_symbols_file(path, unk_string):
    """Get token types from the file containing all valid symbols."""
    with path.open() as fin:
        return get_token_types(
            (
                token
                for line in fin
                for pos_json in json.loads(line)
                for token in pos_json['s'].split()
            ),
            unk_string
        )

def get_file_names_from_directory(directory, use_next_symbols, use_state_annotations):
    if use_next_symbols:
        next_symbols_files = (directory / 'next-symbols.jsonl', directory / 'next-symbols.prepared')
    else:
        next_symbols_files = None
    
    if use_state_annotations:
        states_files = (None, directory / 'states.prepared') # No source for states
    else:
        states_files = None

    return (
        (directory / 'main.tok', directory / 'main.prepared'),
        (directory / 'labels.txt', directory / 'labels.prepared'),
        next_symbols_files,
        states_files
    )

def prepare_labels_file(pair):
    input_path, output_path = pair
    print(f'preparing {input_path} => {output_path}', file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as fin:
        data = []
        for line_no, line in enumerate(fin, 1):
            try:
                label = int(line.strip())
                if label not in (0, 1):
                    raise ValueError(f'expected 0 or 1, got {label}')
            except ValueError as e:
                raise ValueError(f"{input_path}:{line_no}: invalid label: {e}")
            else:
                data.append(bool(label))
    torch.save(data, output_path)

def prepare_valid_symbols_file(vocab, eos_index, pair):
    input_path, output_path = pair
    print(f'preparing {input_path} => {output_path}', file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as fin:
        data = []
        for line_no, line in enumerate(fin, 1):
            line_data = []
            line_json = json.loads(line)
            for pos_json in line_json:
                try:
                    pos_data = [vocab.to_int(t) for t in pos_json['s'].split()]
                except KeyError as e:
                    raise ValueError(f'{input_path}:{line_no}: invalid token: {e}')
                if pos_json['e']:
                    pos_data.append(eos_index)
                line_data.append(pos_data)
            data.append(line_data)
    torch.save(data, output_path)

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description='Convert tokenized text to a prepared, integerized form.')
    parser.add_argument('--training-data', type=pathlib.Path, required=True, help='Directory with training data (main.tok, labels.txt).')
    parser.add_argument('--more-data', action='append', default=[], help='Additional datasets to prepare using the training vocab.')
    parser.add_argument('--always-allow-unk', action='store_true', default=False)
    parser.add_argument('--never-allow-unk', action='store_true', default=False)
    parser.add_argument('--use-next-symbols', action='store_true', default=False)
    parser.add_argument('--only-more-data', action='store_true', default=False)
    
    # Add arguments for FST annotation
    parser.add_argument('--use-state-annotations', action='store_true', help='Enable FST-based state annotations.')
    parser.add_argument('--fst-annotator-path', type=pathlib.Path, help='Path to the FST data file for annotation.')

    add_prepare_data_args(parser)
    args = parser.parse_args()
    validate_prepare_data_args(parser, args)

    if args.always_allow_unk and args.never_allow_unk:
        parser.error('cannot pass both --always-allow-unk and --never-allow-unk')
    if args.use_state_annotations and not args.fst_annotator_path:
        parser.error('--fst-annotator-path is required for --use-state-annotations')

    # Load and reconstruct annotator FST if needed
    annotator = None
    if args.use_state_annotations:
        print("Loading and reconstructing FST annotator...")
        fst_data = torch.load(args.fst_annotator_path, weights_only=False)
        annotator = reconstruct_fst_from_data(fst_data)
        print("FST annotator ready.")

    training_files = get_file_names_from_directory(args.training_data, args.use_next_symbols, args.use_state_annotations)
    prepared_files = []
    if not args.only_more_data:
        prepared_files.append(training_files)
    for arg in args.more_data:
        prepared_files.append(get_file_names_from_directory(args.training_data / 'datasets' / arg, args.use_next_symbols, args.use_state_annotations))

    unk_string = None if args.never_allow_unk else args.unk_string

    # Build vocabulary
    if args.use_next_symbols:
        token_types, has_unk = get_token_types_in_next_symbols_file(training_files[2][0], unk_string)
    elif annotator:
        print("Building vocabulary from annotated tokens...")
        token_types, has_unk = get_annotated_token_types_in_file(training_files[0][0], unk_string, annotator)
    else:
        token_types, has_unk = get_token_types_in_file(training_files[0][0], unk_string)
    
    allow_unk = (args.always_allow_unk or has_unk) and not args.never_allow_unk
    tokens = sorted(token_types)
    vocab = build_softmax_vocab(tokens, allow_unk, ToIntVocabularyBuilder())
    eos_index = build_softmax_vocab(tokens, allow_unk, ToStringVocabularyBuilder()).eos_index

    # Save vocabulary
    vocab_output_file = args.training_data / 'main.vocab'
    print(f'vocabulary size: {len(vocab)}', file=sys.stderr)
    if not args.only_more_data:
        print(f'writing {vocab_output_file}', file=sys.stderr)
        vocab_output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'tokens': tokens, 'allow_unk': allow_unk}, vocab_output_file)

    # Prepare all specified datasets
    for strings_files, labels_files, next_symbols_files, states_files in prepared_files:
        if annotator:
            prepare_annotated_file_and_states(vocab, annotator, strings_files, states_files)
        else:
            prepare_file(vocab, strings_files)
        
        prepare_labels_file(labels_files)
        if args.use_next_symbols:
            prepare_valid_symbols_file(vocab, eos_index, next_symbols_files)

if __name__ == '__main__':
    main()
