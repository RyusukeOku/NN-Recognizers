import argparse
import json
import pathlib
import sys

import torch
from rau.tasks.common.data_preparation import (
    add_prepare_data_args,
    get_token_types_in_file,
    prepare_file,
    validate_prepare_data_args,
    get_token_types
)
from rau.tasks.language_modeling.vocabulary import build_softmax_vocab
from rau.vocab import ToIntVocabularyBuilder, ToStringVocabularyBuilder
from rayuela.fsa.fsa import FSA
from rayuela.fsa.fst import FST
from rayuela.base.semiring import Tropical


def annotate_string(input_string: str, annotator_fst: FST) -> list[str]:
    """Annotates a single string with the FST."""
    tokens = input_string.strip().split()
    if not tokens:
        return []

    input_fsa = FSA.from_string(tokens)
    try:
        composed_fst = annotator_fst.compose(input_fsa)
        best_path = composed_fst.shortest_path(semiring=Tropical)
        if best_path is None:
            return tokens  # Fallback to original tokens
        return best_path.output_string
    except Exception:
        # Fallback for strings not in the FST's language
        return tokens

def annotate_and_get_token_types(path, unk_string, annotator_fst):
    """
    Reads tokens from a file, annotates them using the FST, and returns the set of unique annotated tokens.
    """
    def generate_annotated_tokens():
        with path.open() as fin:
            for line in fin:
                annotated_tokens = annotate_string(line, annotator_fst)
                for token in annotated_tokens:
                    yield token

    return get_token_types(generate_annotated_tokens(), unk_string)


def prepare_annotated_file(vocab, annotator_fst, pair):
    """
    Annotates each string in the input file using the FST and converts it to integer IDs using the vocabulary.
    """
    input_path, output_path = pair
    print(f'preparing annotated tokens in {input_path} => {output_path}', file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as fin, torch.serialization.open_file(str(output_path), 'w') as fout:
        data = []
        for line in fin:
            annotated_tokens = annotate_string(line, annotator_fst)
            try:
                data.append(torch.tensor([vocab.to_int(t) for t in annotated_tokens]))
            except KeyError as e:
                raise ValueError(f'{input_path}: unknown token: {e}')
        torch.save(data, fout)


def get_positional_token_types_in_file(path, unk_string):
    """
    ファイルからトークンを読み込み、位置情報を付加したトークンのセットを返します。
    例: "the cat" -> {"the_0", "cat_1"}
    """
    def generate_positional_tokens():
        with path.open() as fin:
            for line in fin:
                tokens = line.strip().split()
                for i, token in enumerate(tokens):
                    yield f"{token}_{i}" # トークンと位置を結合

    return get_token_types(generate_positional_tokens(), unk_string)


def prepare_positional_file(vocab, pair):
    """
    ファイル内の各トークンに位置情報を付加し、語彙を使って整数に変換します。
    """
    input_path, output_path = pair
    print(f'preparing positional tokens in {input_path} => {output_path}', file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as fin, torch.serialization.open_file(str(output_path), 'w') as fout:
        data = []
        for line in fin:
            tokens = line.strip().split()
            # 各トークンに位置情報を付加
            positional_tokens = [f"{token}_{i}" for i, token in enumerate(tokens)]
            try:
                # 語彙を使って整数IDに変換
                data.append(torch.tensor([vocab.to_int(t) for t in positional_tokens]))
            except KeyError as e:
                # 語彙にないトークンが見つかった場合のエラー
                raise ValueError(f'{input_path}: unknown token: {e}')
        torch.save(data, fout)

def get_token_types_in_next_symbols_file(path, unk_string):
    """
    Get token types from the file containing all valid symbols.
    """
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
    # Each pair is (input file, output file).
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
                raise ValueError(
                    f"{input_path}:{line_no}: in line {line.rstrip()!r}: "
                    f"invalid label: {e}"
                )
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

def main():

    parser = argparse.ArgumentParser(
        description=
        'Convert tokenized text to a prepared, integerized form that can be '
        'loaded efficiently. Input files (.tok) should have one sequence of '
        'whitespace-separated tokens per line. A prepared output file '
        '(.prepared) and a vocabulary file (.vocab) will be written.'
    )
    parser.add_argument('--training-data', type=pathlib.Path, required=True,
        help='A directory containing training data. The file '
             '<training-data>/main.tok will be used as input, and the file '
             '<training-data>/main.prepared will be used as output. '
             'The vocabulary will be saved to the file '
             '<training-data>/main.vocab.')
    parser.add_argument('--more-data', action='append', default=[],
        help='Name of an additional dataset in the training data directory '
             'that will be prepared using the training data. This option can '
             'be passed multiple times. The file '
             '<training-data>/datasets/<more-data>/main.tok will be used as '
             'input, and the file '
             '<training-data>/datasets/<more-data>/main.prepared will be used '
             'as output.')
    parser.add_argument('--always-allow-unk', action='store_true', default=False,
        help='Always allow the vocabulary to include an <unk> token, even if '
             'one does not appear in the training data.')
    parser.add_argument('--never-allow-unk', action='store_true', default=False,
        help='Never allow the vocabulary to include an <unk> token; treat '
             'every token as a normal token in the vocabulary. This is useful '
             'for datasets that already have <unk> preprocessing done.')
    parser.add_argument('--use-next-symbols', action='store_true', default=False,
        help='Whether to prepare an additional file (.jsonl) of valid next '
             'symbols at each position in the string for each of the '
             'training, test, and validation datasets.')
    parser.add_argument('--only-more-data', action='store_true', default=False,
        help='Do not write the output files for the training data and '
             'vocabulary. Only process the additional datasets.')
    parser.add_argument('--use-state-annotations', action='store_true', default=False,
                        help='Annotate input strings with FST states.')
    parser.add_argument('--fst-annotator-path', type=pathlib.Path,
                        help='Path to the FST for annotation. Required if --use-state-annotations is set.')
    add_prepare_data_args(parser)
    args = parser.parse_args()
    validate_prepare_data_args(parser, args)

    if args.always_allow_unk and args.never_allow_unk:
        parser.error('cannot pass both --always-allow-unk and --never-allow-unk')

    training_files = get_file_names_from_directory(
        args.training_data,
        use_next_symbols=args.use_next_symbols
    )
    prepared_files = []
    if not args.only_more_data:
        prepared_files.append(training_files)
    for arg in args.more_data:
        prepared_files.append(
            get_file_names_from_directory(
                args.training_data / 'datasets' / arg,
                use_next_symbols=args.use_next_symbols
            )
        )

    unk_string = None if args.never_allow_unk else args.unk_string

    if args.use_state_annotations:
        if not args.fst_annotator_path:
            parser.error("--fst-annotator-path is required when using --use-state-annotations")
        annotator_fst = FST.read(args.fst_annotator_path)

    if args.use_next_symbols:
        # If we use next symbols data, build the vocabulary from that data,
        # as there can be more symbols than in the strings data.
        token_types, has_unk = get_token_types_in_next_symbols_file(
            training_files[2][0],
            unk_string
        )
    elif args.use_state_annotations:
        token_types, has_unk = annotate_and_get_token_types(
            training_files[0][0],
            unk_string,
            annotator_fst
        )
    else:
        token_types, has_unk = get_token_types_in_file(
            training_files[0][0],
            unk_string
        )
    allow_unk = (args.always_allow_unk or has_unk) and not args.never_allow_unk

    tokens = sorted(token_types)
    vocab = build_softmax_vocab(tokens, allow_unk, ToIntVocabularyBuilder())
    # TODO Make this more efficient.
    eos_index = build_softmax_vocab(tokens, allow_unk, ToStringVocabularyBuilder()).eos_index

    vocab_output_file = args.training_data / 'main.vocab'
    print(f'token types: {len(token_types)}', file=sys.stderr)
    print(f'vocabulary size: {len(vocab)}', file=sys.stderr)
    print(f'has unk ({unk_string}): {has_unk}', file=sys.stderr)
    print(f'allow unk: {allow_unk}', file=sys.stderr)
    if not args.only_more_data:
        print(f'writing {vocab_output_file}', file=sys.stderr)
        vocab_output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'tokens' : tokens,
            'allow_unk' : allow_unk
        }, vocab_output_file)
    for strings_files, labels_files, next_symbols_files in prepared_files:
        if args.use_state_annotations:
            prepare_annotated_file(vocab, annotator_fst, strings_files)
        else:
            prepare_file(vocab, strings_files)
        prepare_labels_file(labels_files)
        if args.use_next_symbols:
            prepare_valid_symbols_file(vocab, eos_index, next_symbols_files)

    # for strings_files, labels_files, next_symbols_files in prepared_files:
    #     # 元のprepare_fileの代わりに、新しく作成した関数を呼び出す
    #     prepare_positional_file(vocab, strings_files)
    #     # 他のファイルの準備は変更なし
    #     prepare_labels_file(labels_files)
    #     if args.use_next_symbols:
    #         # 注意: next_symbolsの準備も位置情報に対応させる必要がありますが、
    #         # ここでは簡単のため、文字列の準備に焦点を当てています。
    #         prepare_valid_symbols_file(vocab, eos_index, next_symbols_files)

if __name__ == '__main__':
    main()
