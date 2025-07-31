
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

def annotate_string(tokens: list[str], annotator_fst: FST) -> list[tuple[str, str]]:
    """Annotates a list of tokens using the provided FST, returning (token, state) pairs."""
    if not tokens:
        return []
    
    # Manually construct a linear FSA from the input tokens
    input_fsa = FSA(R=annotator_fst.R)
    num_states = len(tokens) + 1
    for i in range(num_states):
        input_fsa.add_state(i)
    input_fsa.set_I(0)
    input_fsa.add_F(num_states - 1, annotator_fst.R(0.0)) # Add final state with appropriate weight

    for i, token in enumerate(tokens):
        input_fsa.add_arc(i, token, i + 1, annotator_fst.R(0.0))
    
    try:
        composed_fst = annotator_fst.compose(input_fsa)
        best_path = composed_fst.shortest_path()
        
        if best_path:
            annotated_pairs = []
            for arc in best_path.path: # best_path.path is a list of Arc objects
                original_token = arc.input_symbol
                annotated_token_str = arc.output_symbol
                # Assuming format is {token}_{state}.
                # If state name can contain '_', a more robust parsing or FST output format is needed.
                parts = annotated_token_str.split('_', 1)
                if len(parts) > 1:
                    state_name = parts[1]
                else:
                    state_name = "UNK_STATE" # Fallback if parsing fails
                annotated_pairs.append((original_token, state_name))
            return annotated_pairs
        else:
            return [(token, "UNK_STATE") for token in tokens] # Fallback if no path found
    except Exception as e:
        print(f"Error during FST annotation: {e}", file=sys.stderr)
        return [(token, "UNK_STATE") for token in tokens] # Fallback to original tokens and UNK_STATE on error

def get_annotated_token_types_in_file(path, unk_string, annotator_fst):
    """Reads tokens, annotates them, and returns the set of unique annotated tokens and state names."""
    token_types = set()
    state_types = set()
    def generate_annotated_pairs():
        with path.open() as fin:
            for line in fin:
                tokens = line.strip().split()
                annotated_pairs = annotate_string(tokens, annotator_fst)
                for token, state in annotated_pairs:
                    yield token, state
    
    for token, state in generate_annotated_pairs():
        token_types.add(token)
        state_types.add(state)

    # Add UNK string if applicable
    if unk_string:
        token_types.add(unk_string)
        state_types.add(unk_string) # Assuming UNK state is also possible

    # Ensure state_types is not empty if state annotations are used
    if not state_types:
        state_types.add("UNK_STATE") # Add a default UNK state if no states were found

    return sorted(list(token_types)), sorted(list(state_types))

def prepare_annotated_file(token_vocab, state_vocab, pair, annotator):
    """Annotates strings in a file and saves them as integerized tensors for tokens and states."""
    input_path, output_path = pair
    print(f'preparing annotated tokens and states in {input_path} => {output_path}', file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_data = []
    state_data = []
    with input_path.open() as fin:
        for line in fin:
            tokens = line.strip().split()
            annotated_pairs = annotate_string(tokens, annotator)
            try:
                token_data.append(torch.tensor([token_vocab.to_int(t) for t, _ in annotated_pairs]))
                state_data.append(torch.tensor([state_vocab.to_int(s) for _, s in annotated_pairs]))
            except KeyError as e:
                raise ValueError(f'{input_path}: unknown token or state: {e}')
        torch.save({'tokens': token_data, 'states': state_data}, output_path)

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

def get_file_names_from_directory(directory, use_next_symbols):
    if use_next_symbols:
        next_symbols_files = (directory / 'next-symbols.jsonl', directory / 'next-symbols.prepared')
    else:
        next_symbols_files = None
    return (
        (directory / 'main.tok', directory / 'main.prepared'),
        (directory / 'labels.txt', directory / 'labels.prepared'),
        next_symbols_files
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

    training_files = get_file_names_from_directory(args.training_data, args.use_next_symbols)
    prepared_files = []
    if not args.only_more_data:
        prepared_files.append(training_files)
    for arg in args.more_data:
        prepared_files.append(get_file_names_from_directory(args.training_data / 'datasets' / arg, args.use_next_symbols))

    unk_string = None if args.never_allow_unk else args.unk_string

    # Build vocabulary
    if args.use_next_symbols:
        token_types, has_unk = get_token_types_in_next_symbols_file(training_files[2][0], unk_string)
        state_types = [] # No state types for next symbols
    elif annotator:
        print("Building vocabulary from annotated tokens and states...")
        token_types, state_types = get_annotated_token_types_in_file(training_files[0][0], unk_string, annotator)
        has_unk = unk_string in token_types or unk_string in state_types
    else:
        token_types, has_unk = get_token_types_in_file(training_files[0][0], unk_string)
        state_types = [] # No state types without annotator
    
    allow_unk = (args.always_allow_unk or has_unk) and not args.never_allow_unk
    
    token_vocab = build_softmax_vocab(token_types, allow_unk, ToIntVocabularyBuilder())
    state_vocab = build_softmax_vocab(state_types, allow_unk, ToIntVocabularyBuilder()) if state_types else None
    eos_index = build_softmax_vocab(token_types, allow_unk, ToStringVocabularyBuilder()).eos_index

    # Save vocabulary
    vocab_output_file = args.training_data / 'main.vocab'
    print(f'token vocabulary size: {len(token_vocab)}', file=sys.stderr)
    if state_vocab:
        print(f'state vocabulary size: {len(state_vocab)}', file=sys.stderr)
    if not args.only_more_data:
        print(f'writing {vocab_output_file}', file=sys.stderr)
        vocab_output_file.parent.mkdir(parents=True, exist_ok=True)
        vocab_data = {'tokens': token_types, 'allow_unk': allow_unk}
        if state_types:
            vocab_data['states'] = state_types
        torch.save(vocab_data, vocab_output_file)

    # Prepare all specified datasets
    for strings_files, labels_files, next_symbols_files in prepared_files:
        if annotator:
            prepare_annotated_file(token_vocab, state_vocab, strings_files, annotator)
        else:
            prepare_file(token_vocab, strings_files)
        
        prepare_labels_file(labels_files)
        if args.use_next_symbols:
            prepare_valid_symbols_file(token_vocab, eos_index, next_symbols_files)

if __name__ == '__main__':
    main()
