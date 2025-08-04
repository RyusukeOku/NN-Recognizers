

import argparse
import json
import pathlib
import sys

import torch

# Imports for FST annotation
from rayuela.fsa.fsa import FSA
from rayuela.fsa.fst import FST
from rayuela.base.semiring import Tropical, Real
from collections import defaultdict

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

def find_shortest_path(fst: FST) -> list[str] | None:
    """Finds the shortest path in an FST using a Viterbi-like algorithm."""
    
    # Assumes Tropical semiring for shortest path
    if not issubclass(fst.R, Tropical):
        raise TypeError("Shortest path finding requires a Tropical semiring.")

    dist = defaultdict(lambda: fst.R.zero)
    backpointer = {}

    # Initialization
    initial_state = next(iter(fst.I), None)
    if initial_state is None:
        return None
    dist[initial_state] = fst.R.one

    # Viterbi forward pass using topological sort
    try:
        queue = fst.toposort()
    except Exception as e:
        # Fallback for potential cycles, though compose should be acyclic
        print(f"DEBUG: fst.toposort() failed with {e}. Falling back to simple sort.", file=sys.stderr)
        queue = sorted(list(fst.Q))

    for p in queue:
        if dist[p] == fst.R.zero:
            continue
        for i, o, q, w in fst.arcs(p):
            new_dist = dist[p] * w
            if new_dist < dist[q]:
                dist[q] = new_dist
                backpointer[q] = (p, o)

    # Find the best final state
    best_final_state = None
    min_dist = fst.R.zero

    for f_state in fst.F:
        final_weight = dist[f_state] * fst.F[f_state]
        if final_weight < fst.R.zero:
            if best_final_state is None or final_weight < min_dist:
                min_dist = final_weight
                best_final_state = f_state

    if best_final_state is None:
        return None

    # Backtracking
    path = []
    curr = best_final_state
    while curr in backpointer:
        prev, output_sym = backpointer[curr]
        path.append(output_sym)
        curr = prev
    
    path.reverse()
    return path

def annotate_string(tokens: list[str], annotator_fst: FST) -> list[str]:
    """Annotates a list of tokens using the provided FST."""
    if not tokens:
        return []
    
    # Manually construct a linear FSA from the input tokens
    input_fsa = FSA(R=annotator_fst.R)
    num_states = len(tokens) + 1
    for i in range(num_states):
        input_fsa.add_state(i)
    input_fsa.set_I(0)
    input_fsa.add_F(num_states - 1, annotator_fst.R(0.0))

    for i, token in enumerate(tokens):
        input_fsa.add_arc(i, token, i + 1, annotator_fst.R(0.0))
    
    # Manually convert the input FSA to an FST for composition
    input_fst = FST(R=input_fsa.R)
    for state in input_fsa.Q:
        input_fst.add_state(state)
    input_fst.set_I(next(iter(input_fsa.I)))
    for final_state in input_fsa.F:
        input_fst.add_F(final_state, input_fsa.R(0.0))
    for p in input_fsa.Q:
        for i, q, w in input_fsa.arcs(p):
            input_fst.add_arc(p, i, i, q, w)

    try:
        composed_fst = annotator_fst.compose(input_fst)
        shortest_path_tokens = find_shortest_path(composed_fst)
        
        if shortest_path_tokens is None:
            print(f"DEBUG: No shortest path found for tokens: {tokens}", file=sys.stderr)
            return tokens
        return shortest_path_tokens

    except Exception as e:
        print(f"ERROR: Exception during FST processing for tokens '{tokens}': {e}", file=sys.stderr)
        return tokens # Fallback to original tokens on error

def get_annotated_token_types_in_file(path, unk_string, annotator_fst):
    """Reads tokens, annotates them, and returns the set of unique annotated tokens."""
    def generate_annotated_tokens():
        with path.open() as fin:
            for line in fin:
                tokens = line.strip().split()
                annotated_tokens = annotate_string(tokens, annotator_fst)
                for token in annotated_tokens:
                    yield token
    return get_token_types(generate_annotated_tokens(), unk_string)

def prepare_annotated_file(vocab, annotator_fst, pair, text_output_file=None):
    """Annotates strings in a file and saves them as integerized tensors."""
    input_path, output_path = pair
    print(f'preparing annotated tokens in {input_path} => {output_path}', file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as fin:
        data = []
        for line in fin:
            tokens = line.strip().split()
            annotated_tokens = annotate_string(tokens, annotator_fst)
            if text_output_file:
                text_output_file.write(' '.join(annotated_tokens) + '\n')
            try:
                data.append(torch.tensor([vocab.to_int(t) for t in annotated_tokens]))
            except KeyError as e:
                raise ValueError(f'{input_path}: unknown token: {e}')
        torch.save(data, output_path)

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
    parser.add_argument('--annotated-text-output-path', type=pathlib.Path, help='Path to save the annotated tokens in text format for inspection.')

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
        print("Loading and reconstructing FST annotator...", file=sys.stderr)
        fst_data = torch.load(args.fst_annotator_path, weights_only=False)
        annotator = reconstruct_fst_from_data(fst_data)
        print("FST annotator ready.", file=sys.stderr)

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
    elif annotator:
        print("Building vocabulary from annotated tokens...", file=sys.stderr)
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
    annotated_text_output_file = None
    if args.annotated_text_output_path:
        args.annotated_text_output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_text_output_file = args.annotated_text_output_path.open('w')

    for strings_files, labels_files, next_symbols_files in prepared_files:
        if annotator:
            prepare_annotated_file(vocab, annotator, strings_files, annotated_text_output_file)
        else:
            prepare_file(vocab, strings_files)
        
        prepare_labels_file(labels_files)
        if args.use_next_symbols:
            prepare_valid_symbols_file(vocab, eos_index, next_symbols_files)
    
    if annotated_text_output_file:
        annotated_text_output_file.close()


if __name__ == '__main__':
    main()
