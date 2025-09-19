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
    
    # For inspection purposes, we don't need special BOS/EOS handling
    builder = ToIntVocabularyBuilder()
    vocab = build_softmax_vocab(
        tokens,
        allow_unk,
        builder,
        use_bos=False,
        use_eos=False
    )
    return vocab, tokens

def main():
    parser = argparse.ArgumentParser(description='Learn and inspect an automaton from data using RPNI.')
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
        vocab, token_list = load_vocab(args.vocab_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Learning automaton from {args.data_dir} with RPNI...")
    rpni_learner = RPNILearner.from_files(main_tok_path, labels_txt_path, vocab)
    container = rpni_learner.learn()
    print("--- RPNI Learned Automaton ---")
    
    num_states = container.num_states()
    print(f"Number of States: {num_states}")
    
    initial_state = container.initial_state()
    print(f"Initial State: {initial_state}")

    final_states = [i for i in range(num_states) if container.is_accept_state(State(i))]
    print(f"Final States: {final_states}")

    print("Transitions:")
    # Sort transitions for consistent and readable output
    sorted_transitions = sorted(list(container.transitions()), key=lambda t: (t.state_from, t.symbol))
    for t in sorted_transitions:
        # Map symbol ID back to string representation
        symbol_str = token_list[t.symbol]
        print(f"  ({t.state_from}) --'{symbol_str}'--> ({t.state_to})")

if __name__ == '__main__':
    main()
