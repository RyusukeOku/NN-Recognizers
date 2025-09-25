import argparse
import pathlib
import torch

from recognizers.automata.rpni_learner import RPNILearner
from rau.vocab import Vocabulary, ToIntVocabularyBuilder
from rau.tasks.language_modeling.vocabulary import build_softmax_vocab
from rayuela.base.state import State

def load_vocab(vocab_path: pathlib.Path) -> tuple[Vocabulary, list[str]]:
    """Loads vocabulary data and constructs a Vocabulary object."""
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    vocab_data = torch.load(vocab_path)
    tokens = vocab_data['tokens']
    allow_unk = vocab_data['allow_unk']
    
    builder = ToIntVocabularyBuilder()
    vocab = build_softmax_vocab(
        tokens,
        allow_unk,
        builder
    )
    return vocab, tokens

def main():
    parser = argparse.ArgumentParser(description='Evaluate a learned RPNI automaton.')
    parser.add_argument('--data-dir', type=pathlib.Path, required=True, help='Directory with main.tok and labels.txt')
    parser.add_argument('--vocab-path', type=pathlib.Path, required=True, help='Path to the main.vocab file.')
    args = parser.parse_args()

    main_tok_path = args.data_dir / 'main.tok'
    labels_txt_path = args.data_dir / 'labels.txt'

    if not main_tok_path.is_file() or not labels_txt_path.is_file():
        print(f"Error: main.tok or labels.txt not found in {args.data_dir}")
        return

    try:
        print(f"Loading vocabulary from {args.vocab_path}...")
        vocab, _ = load_vocab(args.vocab_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Learning automaton from {args.data_dir} with RPNI...")
    # We still need to run the learning process to get the automaton
    rpni_learner = RPNILearner.from_files(main_tok_path, labels_txt_path, vocab)
    automaton = rpni_learner.learn()
    
    print(f"\nEvaluating automaton for: {args.data_dir.name}")

    # Build a transition dictionary for efficient lookup
    transitions_dict = {}
    for t in automaton.transitions():
        transitions_dict[(t.state_from, t.symbol)] = t.state_to

    correct_predictions = 0
    total_strings = 0

    # Re-read the files to ensure correct sample-label correspondence
    with open(main_tok_path, 'r') as f_tok, open(labels_txt_path, 'r') as f_labels:
        for line_tok, line_label in zip(f_tok, f_labels):
            total_strings += 1
            
            tokens = line_tok.strip().split()
            true_label = line_label.strip() == 'True'
            
            sample_indices = [vocab.stoi(token) for token in tokens]

            current_state = automaton.initial_state
            is_accepted = True
            
            for symbol_id in sample_indices:
                next_state = transitions_dict.get((current_state, symbol_id))
                if next_state is None:
                    is_accepted = False
                    break
                current_state = next_state
            
            if is_accepted:
                prediction = automaton.is_accept_state(current_state)
            else:
                prediction = False

            if prediction == true_label:
                correct_predictions += 1

    accuracy = (correct_predictions / total_strings) * 100 if total_strings > 0 else 0
    
    print(f"Correct predictions: {correct_predictions} / {total_strings}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
