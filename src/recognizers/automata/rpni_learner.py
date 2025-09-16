import pathlib
from typing import Union, List, Tuple

from src.recognizers.automata.RPNI import run_RPNI
from src.recognizers.automata.finite_automaton import FiniteAutomatonContainer, FiniteAutomatonTransition
from rayuela.base.state import State
from rau.vocab import Vocabulary
from src.recognizers.automata.Dfa import Dfa


class RPNILearner:
    """
    Implements the RPNI algorithm to learn a DFA and represent it as a FiniteAutomatonContainer.
    """

    def __init__(self,
                 positive_examples: List[Tuple[str, ...]],
                 negative_examples: List[Tuple[str, ...]],
                 vocab: Vocabulary):
        """
        Initializes the RPNILearner.

        Args:
            positive_examples: A list of token sequences that should be accepted.
            negative_examples: A list of token sequences that should be rejected.
            vocab: The vocabulary for converting tokens to integer IDs.
        """
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        self.vocab = vocab

    @classmethod
    def from_files(cls, main_tok_path: Union[str, pathlib.Path],
                   label_txt_path: Union[str, pathlib.Path],
                   vocab: Vocabulary) -> "RPNILearner":
        """
        Creates an RPNILearner instance from token and label files.
        """
        main_tok_path = pathlib.Path(main_tok_path)
        label_txt_path = pathlib.Path(label_txt_path)

        with main_tok_path.open('r', encoding='utf-8') as f_tok, label_txt_path.open('r', encoding='utf-8') as f_lbl:
            lines_tok = f_tok.readlines()
            lines_lbl = f_lbl.readlines()

        if len(lines_tok) != len(lines_lbl):
            raise ValueError("Number of lines in token file and label file do not match.")

        positive_examples = []
        negative_examples = []

        for tok_line, lbl_line in zip(lines_tok, lines_lbl):
            tokens = tuple(tok_line.strip().split())
            label = int(lbl_line.strip())
            if label == 1:
                positive_examples.append(tokens)
            else:
                negative_examples.append(tokens)

        return cls(positive_examples, negative_examples, vocab)

    def _prepare_data_for_rpni(self) -> list:
        """
        Formats the examples into the list format required by aalpy's RPNI.
        [[(input_sequence, label)], ...]
        """
        data = []
        for seq in self.positive_examples:
            data.append((seq, True))
        for seq in self.negative_examples:
            data.append((seq, False))
        return data

    def learn(self, algorithm='gsm', print_info=False) -> FiniteAutomatonContainer:
        """
        Runs the RPNI algorithm and returns the learned automaton as a FiniteAutomatonContainer.
        """
        rpni_data = self._prepare_data_for_rpni()

        learned_dfa = run_RPNI(rpni_data, automaton_type='dfa', algorithm=algorithm, print_info=print_info)

        if learned_dfa is None:
            raise RuntimeError("RPNI learning failed. The data might be non-deterministic.")

        return self._convert_aalpy_dfa_to_container(learned_dfa)

    def _convert_aalpy_dfa_to_container(self, dfa: Dfa) -> FiniteAutomatonContainer:
        """
        Converts an aalpy.automata.Dfa object to a FiniteAutomatonContainer.
        """
        aalpy_states = sorted(dfa.states, key=lambda s: s.state_id)
        state_map = {state: i for i, state in enumerate(aalpy_states)}

        initial_state_id = state_map[dfa.initial_state]

        container = FiniteAutomatonContainer(
            num_states=len(aalpy_states),
            alphabet_size=len(self.vocab),
            initial_state=State(initial_state_id)
        )

        for aalpy_state, state_id in state_map.items():
            if aalpy_state.is_accepting:
                container.add_accept_state(State(state_id))

            for symbol_str, target_aalpy_state in aalpy_state.transitions.items():
                try:
                    symbol_id = self.vocab.to_int(symbol_str)
                except KeyError:
                    print(f"Warning: Symbol '{symbol_str}' from learned DFA not in vocabulary. Skipping transition.")
                    continue

                target_state_id = state_map[target_aalpy_state]

                transition = FiniteAutomatonTransition(
                    state_from=State(state_id),
                    symbol=symbol_id,
                    state_to=State(target_state_id)
                )
                container.add_transition(transition)

        return container
