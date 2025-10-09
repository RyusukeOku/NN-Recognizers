import argparse
import logging
import sys
import pathlib

import humanfriendly

from rau.tools.torch.profile import get_current_memory

from recognizers.neural_networks.data import (
    add_data_arguments,
    load_prepared_data,
    load_vocabulary_data
)
from recognizers.neural_networks.model_interface import RecognitionModelInterface
from recognizers.neural_networks.training_loop import (
    RecognitionTrainingLoop,
    add_training_loop_arguments,
    get_training_loop_kwargs,
)
# Imports for L*
from recognizers.learning_algs.SUL import SUL
from recognizers.learning_algs.Oracle import Oracle
from recognizers.learning_algs.LStar import run_Lstar
from recognizers.automata.finite_automaton import FiniteAutomatonContainer, Transition
from rau.vocab import ToIntVocabularyBuilder


class DataSUL(SUL):
    """A System Under Learning for DFA learning from labeled data."""
    def __init__(self, positive_examples):
        super().__init__()
        self.positive_examples = set(positive_examples)

    def pre(self):
        pass

    def post(self):
        pass

    def step(self, letter):
        # L* for DFA uses step(None) for the empty string query.
        if letter is None:
            return tuple() in self.positive_examples
        # This SUL is only for whole-sequence queries, so individual steps are not supported.
        raise NotImplementedError("Step-by-step execution is not supported.")

    def query(self, word: tuple) -> list:
        self.num_queries += 1
        self.num_steps += len(word)
        is_accepted = word in self.positive_examples
        # The observation table for DFA learning expects a list with a single boolean.
        return [is_accepted]

class DataEquivalenceOracle(Oracle):
    """An Equivalence Oracle that finds counterexamples from a finite dataset."""
    def __init__(self, alphabet, sul, all_examples):
        super().__init__(alphabet, sul)
        self.all_examples = all_examples

    def find_cex(self, hypothesis):
        for sequence, is_positive in self.all_examples:
            # The `execute_sequence` method returns the output for each step. For a DFA,
            # the output of the last step is the acceptance value.
            if sequence:
                hypothesis_accepts = hypothesis.execute_sequence(hypothesis.initial_state, sequence)[-1]
            else:
                # Handle the empty sequence separately. Acceptance is determined by the initial state.
                hypothesis_accepts = hypothesis.initial_state.is_accepting

            if hypothesis_accepts != is_positive:
                # Found a discrepancy between the hypothesis and the ground truth data.
                return sequence
        # No counterexample found in the provided data.
        return None


def main():

    # Configure logging to stdout.
    console_logger = logging.getLogger('main')
    console_logger.addHandler(logging.StreamHandler(sys.stdout))
    console_logger.setLevel(logging.INFO)
    console_logger.info(f'arguments: {sys.argv}')

    model_interface = RecognitionModelInterface()

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description=
        'Train a recognizer.'
    )
    add_data_arguments(parser)
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_training_loop_arguments(parser)
    parser.add_argument('--learn-fsa-with-rpni', action='store_true', default=False,
                        help='Learn an FSA with RPNI from the training data and use it in the model.')
    parser.add_argument('--learn-fsa-with-lstar', action='store_true', default=False,
                        help='Learn an FSA with L* from the training data and use it in the model.')
    args = parser.parse_args()
    model_interface.set_attributes_from_args(args)
    console_logger.info(f'parsed arguments: {args}')

    # Are we training on CPU or GPU?
    device = model_interface.get_device(args)
    console_logger.info(f'device: {device}')
    do_profile_memory = device.type == 'cuda'

    # Configure the training loop.
    training_loop = RecognitionTrainingLoop(
        **get_training_loop_kwargs(parser, args)
    )

    # Load the tokens in the vocabulary. This determines the sizes of the
    # embedding and softmax layers in the model.
    vocabulary_data = load_vocabulary_data(args, parser)

    if do_profile_memory:
        memory_before = get_current_memory(device)

    fsa_container = None
    fsa_alphabet = None
    if args.learn_fsa_with_rpni:
        console_logger.info('Learning FSA with RPNI...')
        from recognizers.automata.rpni_learner import RPNILearner

        # Create the vocabulary object needed for RPNI learner
        # This replicates the logic from load_prepared_data -> get_vocabularies
        vocab, _ = model_interface.get_vocabularies(
            vocabulary_data,
            builder=ToIntVocabularyBuilder()
        )
        fsa_alphabet = vocabulary_data.tokens

        # RPNILearner needs paths to main.tok and labels.txt
        main_tok_path = args.training_data / 'main.tok'
        labels_txt_path = args.training_data / 'labels.txt'

        rpni_learner = RPNILearner.from_files(main_tok_path, labels_txt_path, vocab)
        fsa_container = rpni_learner.learn()
        console_logger.info(f'RPNI learned an FSA with {fsa_container.num_states()} states.')
    elif args.learn_fsa_with_lstar:
        console_logger.info('Learning FSA with L*...')

        # Load vocabulary to map tokens to integers.
        vocab, _ = model_interface.get_vocabularies(
            vocabulary_data,
            builder=ToIntVocabularyBuilder()
        )
        fsa_alphabet = vocabulary_data.tokens

        # Load positive and negative examples from data files.
        main_tok_path = args.training_data / 'main.tok'
        labels_txt_path = args.training_data / 'labels.txt'
        positive_examples = set()
        all_examples_for_oracle = []
        with main_tok_path.open() as f_tok, labels_txt_path.open() as f_lbl:
            for line_tok, line_lbl in zip(f_tok, f_lbl):
                tokens = line_tok.strip().split()
                # Convert token strings to a tuple of integers.
                int_sequence = tuple(vocab.to_int(t) for t in tokens)
                is_positive = bool(int(line_lbl.strip()))

                if is_positive:
                    positive_examples.add(int_sequence)
                all_examples_for_oracle.append((int_sequence, is_positive))

        # The alphabet for L* is the set of integer token IDs.
        # The vocabulary object used here does not expose pad_index, so we pass the
        # entire range of token IDs to the learner. The learner will correctly
        # infer that sequences containing padding tokens are not part of the language.
        lstar_alphabet = list(range(len(vocab)))

        # Instantiate the SUL and the Equivalence Oracle.
        sul = DataSUL(positive_examples=positive_examples)
        eq_oracle = DataEquivalenceOracle(
            alphabet=lstar_alphabet,
            sul=sul,
            all_examples=all_examples_for_oracle
        )

        # Run the L* algorithm to learn a DFA.
        # Caching is disabled because the default cache implementation assumes a Mealy machine-style
        # output for each input symbol, which is not true for our DFA-style SUL that provides
        # a single acceptance output for the whole sequence. This mismatch causes incorrect
        # non-determinism errors.
        learned_dfa = run_Lstar(
            alphabet=lstar_alphabet,
            sul=sul,
            eq_oracle=eq_oracle,
            automaton_type='dfa',
            print_level=2,  # Log hypothesis size and final results.
            cache_and_non_det_check=False
        )

        console_logger.info(f'L* learned an FSA with {learned_dfa.size} states.')

        # Convert the learned aalpy DFA to a FiniteAutomatonContainer.
        state_to_id = {state: i for i, state in enumerate(learned_dfa.states)}
        transitions = []
        for state_from, state_id_from in state_to_id.items():
            for symbol, state_to in state_from.transitions.items():
                transitions.append(Transition(
                    state_from=state_id_from,
                    symbol=symbol,
                    state_to=state_to_id[state_to]
                ))

        initial_state_id = state_to_id[learned_dfa.initial_state]
        accepting_states_ids = {state_to_id[s] for s in learned_dfa.states if s.is_accepting}

        fsa_container = FiniteAutomatonContainer(
            initial_state=initial_state_id,
            accepting_states=list(accepting_states_ids),
            transitions=transitions
        )

    # Construct the model.
    saver = model_interface.construct_saver(
        args,
        vocabulary_data,
        fsa_container=fsa_container,
        fsa_alphabet=fsa_alphabet
    )
    # Log some information about the model: parameter random seed, number of
    # parameters, GPU memory.
    if model_interface.parameter_seed is not None:
        console_logger.info(f'parameter random seed: {model_interface.parameter_seed}')
    num_parameters = sum(p.numel() for p in saver.model.parameters())
    console_logger.info(f'number of parameters: {num_parameters}')
    if do_profile_memory:
        model_size_in_bytes = get_current_memory(device) - memory_before
        console_logger.info(f'model size: {humanfriendly.format_size(model_size_in_bytes)}')
    else:
        model_size_in_bytes = None

    # Load the data.
    training_data, validation_data, vocabulary \
        = load_prepared_data(args, parser, vocabulary_data, model_interface)

    # Start logging events to disk.
    with saver.logger() as event_logger:
        event_logger.log('model_info', dict(
            parameter_seed=model_interface.parameter_seed,
            size_in_bytes=model_size_in_bytes,
            num_parameters=num_parameters
        ))
        event_logger.log('training_info', dict(
            max_tokens_per_batch=args.max_tokens_per_batch,
            language_modeling_loss_coefficient=args.language_modeling_loss_coefficient,
            next_symbols_loss_coefficient=args.next_symbols_loss_coefficient
        ))
        # Run the training loop.
        training_loop.run(
            saver,
            model_interface,
            training_data,
            validation_data,
            vocabulary,
            console_logger,
            event_logger
        )

if __name__ == '__main__':
    main()