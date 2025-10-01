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
    parser = argparse.ArgumentParser(description='Evaluate a learned RPNI automaton on a separate dataset.')
    parser.add_argument('--train-data-dir', type=pathlib.Path, required=True, help='Directory with training data (main.tok and labels.txt)')
    parser.add_argument('--eval-data-dir', type=pathlib.Path, required=True, help='Directory with evaluation data (main.tok and labels.txt)')
    parser.add_argument('--vocab-path', type=pathlib.Path, required=True, help='Path to the main.vocab file (used for both training and eval).')
    args = parser.parse_args()

    # --- Setup paths for training and evaluation ---
    train_main_tok_path = args.train_data_dir / 'main.tok'
    train_labels_txt_path = args.train_data_dir / 'labels.txt'
    eval_main_tok_path = args.eval_data_dir / 'main.tok'
    eval_labels_txt_path = args.eval_data_dir / 'labels.txt'

    # --- Validate file existence ---
    if not train_main_tok_path.is_file() or not train_labels_txt_path.is_file():
        print(f"Error: Training data not found in {args.train_data_dir}")
        return
    if not eval_main_tok_path.is_file() or not eval_labels_txt_path.is_file():
        print(f"Error: Evaluation data not found in {args.eval_data_dir}")
        return

    # --- Load vocabulary ---
    try:
        print(f"Loading vocabulary from {args.vocab_path}...")
        vocab, _ = load_vocab(args.vocab_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- Learn automaton from training data ---
    print(f"Learning automaton from {args.train_data_dir} with RPNI...")
    rpni_learner = RPNILearner.from_files(train_main_tok_path, train_labels_txt_path, vocab)
    automaton = rpni_learner.learn()
    
    # --- Evaluate automaton on evaluation data ---
    print(f"\nEvaluating automaton on: {args.eval_data_dir.name}")

    # Build a transition dictionary for efficient lookup
    transitions_dict = {}
    for t in automaton.transitions():
        transitions_dict[(t.state_from, t.symbol)] = t.state_to

    correct_predictions = 0
    total_strings = 0

    # Read the evaluation files to ensure correct sample-label correspondence
    with open(eval_main_tok_path, 'r') as f_tok, open(eval_labels_txt_path, 'r') as f_labels:
        for line_tok, line_label in zip(f_tok, f_labels):
            total_strings += 1
            
            tokens = line_tok.strip().split()
            true_label = line_label.strip() == 'True'
            
            try:
                sample_indices = [vocab.to_int(token) for token in tokens]
            except KeyError as e:
                print(f"Warning: Token {e} not in vocabulary. Skipping string: '{' '.join(tokens)}'")
                continue

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
