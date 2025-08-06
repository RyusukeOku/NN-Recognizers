
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

from rayuela.base.state import State # Added import
from rayuela.base.symbol import Sym # Added import

# --- Functions for FST State Annotation ---

from rayuela.base.state import State # Added import
from rayuela.base.symbol import Sym # Added import
from rayuela.fsa.pathsum import Pathsum, Strategy # Added import

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
    for state_idx in fst_data['states']:
        fst.add_state(State(state_idx))
    
    if fst_data['initial_state'] is not None:
        fst.set_I(State(fst_data['initial_state']))
    
    for final_state_idx in fst_data['final_states']:
        fst.add_F(State(final_state_idx), semiring(0.0))
    
    for p_idx, i_sym, o_sym, q_idx, w_val in fst_data['arcs']:
        fst.add_arc(State(p_idx), Sym(i_sym), Sym(o_sym), State(q_idx), semiring(w_val))
        
    return fst

def annotate_string(tokens: list[str], annotator_fst: FST) -> list[str]:
    """Annotates a list of tokens using the provided FST."""
    if not tokens:
        return []
    
    # Manually construct a linear FST from the input tokens
    input_fst = FST(R=annotator_fst.R)
    num_states = len(tokens) + 1
    for i in range(num_states):
        input_fst.add_state(State(i))
    input_fst.set_I(State(0))
    input_fst.add_F(State(num_states - 1), annotator_fst.R(0.0)) # Add final state with appropriate weight

    for i, token in enumerate(tokens):
        # FST requires both input and output symbols. Using token for both for identity.
        input_fst.add_arc(State(i), Sym(token), Sym(token), State(i + 1), annotator_fst.R(0.0))
    
    print(f"DEBUG: Type of annotator_fst: {type(annotator_fst)}")
    print(f"DEBUG: Type of input_fst: {type(input_fst)}")

    try:
        composed_fst = input_fst._compose(annotator_fst)

        print(f"DEBUG: Composed FST after _compose:")
        print(f"  Num states: {composed_fst.num_states}")
        print(f"  Initial states: {list(composed_fst.I)}")
        print(f"  Final states: {list(composed_fst.F)}")
        print(f"  Arcs:")
        for q in composed_fst.Q:
            for arc_info in composed_fst.arcs(q):
                print(f"    {q} --{arc_info[0]}--> {arc_info[1]} (weight: {arc_info[2]})", file=sys.stderr)
        print("--------------------------------------------------", file=sys.stderr)

        # Use Pathsum to find the shortest path
        pathsum_obj = Pathsum(composed_fst)
        best_path_semiring = pathsum_obj.pathsum(Strategy.VITERBI) # Viterbi for shortest path in Tropical semiring
        print(f"DEBUG: Best path semiring value: {best_path_semiring}")
        # Always use a greedy forward pass to annotate the longest possible prefix.
        # This simplifies the logic and provides consistent annotation behavior
        # for both accepted and rejected strings.
        annotated_tokens = []
        try:
            # Assume a single initial state for simplicity
            curr_state = next(composed_fst.I)[0]
            
            for i, token_str in enumerate(tokens):
                token_sym = Sym(token_str)
                
                found_arc = False
                # Find the first arc that matches the current token
                # Note: This is a greedy choice. If multiple arcs match the token,
                # it takes the first one the FST provides.
                for (in_sym, out_sym), next_state, arc_weight in composed_fst.arcs(curr_state):
                    if in_sym == token_sym:
                        annotated_tokens.append(str(out_sym))
                        curr_state = next_state
                        found_arc = True
                        break
                
                if not found_arc:
                    # We are stuck. Append the rest of the original tokens and stop.
                    annotated_tokens.extend(tokens[i:])
                    break
            
            return annotated_tokens

        except StopIteration:
            # No initial state in composed_fst, return original tokens.
            return tokens

    except Exception as e:
        # Added detailed error logging
        print(f"ERROR in annotate_string: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
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

def prepare_annotated_file(vocab, annotator_fst, pair):
    """Annotates strings in a file and saves them as integerized tensors."""
    input_path, output_path = pair
    print(f'preparing annotated tokens in {input_path} => {output_path}', file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as fin:
        data = []
        for line in fin:
            tokens = line.strip().split()
            annotated_tokens = annotate_string(tokens, annotator_fst)
            try:
                print(annotated_tokens)
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

    # Build vocabulary from all datasets
    if annotator:
        print("Building vocabulary from annotated tokens across all datasets...")
        all_token_types = set()
        all_has_unk = False
        for strings_files, _, _ in prepared_files:
            types, has_unk = get_annotated_token_types_in_file(strings_files[0], unk_string, annotator)
            all_token_types.update(types)
            if has_unk:
                all_has_unk = True
        token_types, has_unk = all_token_types, all_has_unk
    elif args.use_next_symbols:
        token_types, has_unk = get_token_types_in_next_symbols_file(training_files[2][0], unk_string)
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
    for strings_files, labels_files, next_symbols_files in prepared_files:
        if annotator:
            prepare_annotated_file(vocab, annotator, strings_files)
        else:
            prepare_file(vocab, strings_files)
        
        prepare_labels_file(labels_files)
        if args.use_next_symbols:
            prepare_valid_symbols_file(vocab, eos_index, next_symbols_files)

if __name__ == '__main__':
    main()
