import pathlib
from typing import Union, List, Tuple

# EDSMアルゴリズムをインポート
from src.recognizers.automata.GsmAlgorithms import run_EDSM
from src.recognizers.automata.automaton_aalpy import DeterministicAutomaton
from src.recognizers.automata.finite_automaton import FiniteAutomatonContainer, FiniteAutomatonTransition
from rayuela.base.state import State
from rau.vocab import Vocabulary
# Dfa クラスの代わりに DeterministicAutomaton をインポート
# (run_EDSM の戻り値の型アノテーションに基づき)

class EDSMLearner:
    """
    Implements the EDSM algorithm to learn a DFA and represent it as a FiniteAutomatonContainer.
    (Based on RPNILearner structure)
    """

    def __init__(self,
                 learning_data: List[Tuple[Tuple[str, ...], bool]],
                 vocab: Vocabulary):
        """
        Initializes the EDSMLearner.

        Args:
            learning_data: A list of (token_sequence, label) tuples.
            vocab: The vocabulary for converting tokens to integer IDs.
        """
        self.learning_data = learning_data
        self.vocab = vocab
        self._alphabet_list = None # アルファベットをキャッシュするため

    @classmethod
    def from_files(cls, main_tok_path: Union[str, pathlib.Path],
                   label_txt_path: Union[str, pathlib.Path],
                   vocab: Vocabulary) -> "EDSMLearner":
        """
        Creates an EDSMLearner instance from token and label files.
        """
        main_tok_path = pathlib.Path(main_tok_path)
        label_txt_path = pathlib.Path(label_txt_path)

        learning_data: List[Tuple[Tuple[str, ...], bool]] = []
        alphabet_set = set()

        with main_tok_path.open() as f_tok, label_txt_path.open() as f_labels:
            for line_tok, line_label in zip(f_tok, f_labels):
                tokens = tuple(line_tok.strip().split())
                label = bool(int(line_label.strip()))
                learning_data.append((tokens, label))
                alphabet_set.update(tokens)
        
        learner = cls(learning_data, vocab)
        learner._alphabet_list = sorted(list(alphabet_set))
        return learner

    def learn(self, delta: float = 0.005) -> FiniteAutomatonContainer:
        """
        Runs the EDSM algorithm and converts the result to a FiniteAutomatonContainer.
        """
        print(f"Running EDSM with delta={delta}...")
        
        # run_EDSM を呼び出す (automaton_type='dfa' を指定)
        learned_dfa: DeterministicAutomaton = run_EDSM(
            data=self.learning_data,
            automaton_type='dfa',
            delta=delta,
            print_info=True # 必要に応じて False に変更
        )

        if learned_dfa is None:
            raise RuntimeError("EDSM learning failed and returned None.")
        
        print(f"Learned AALpy DFA with {len(learned_dfa.states)} states.")
        print("Converting AALpy DFA to FiniteAutomatonContainer...")

        return self._convert_aalpy_dfa_to_container(learned_dfa)

    def get_alphabet(self) -> List[str]:
        """
        Returns the alphabet derived from the training data.
        """
        if self._alphabet_list is None:
            # from_files 以外で初期化された場合
            alphabet_set = set()
            for tokens, _ in self.learning_data:
                alphabet_set.update(tokens)
            self._alphabet_list = sorted(list(alphabet_set))
        
        return self._alphabet_list

    def _convert_aalpy_dfa_to_container(self, dfa: DeterministicAutomaton) -> FiniteAutomatonContainer:
        """
        Converts an aalpy.automata.DeterministicAutomaton object to a FiniteAutomatonContainer.
        (Copied from RPNILearner)
        """
        aalpy_states = sorted(dfa.states, key=lambda s: s.state_id)
        state_map = {state: i for i, state in enumerate(aalpy_states)}

        if dfa.initial_state not in state_map:
             raise ValueError("Learned DFA's initial state not in state list.")

        initial_state_id = state_map[dfa.initial_state]

        container = FiniteAutomatonContainer(
            num_states=len(aalpy_states),
            alphabet_size=len(self.vocab),
            initial_state=initial_state_id
        )

        for aalpy_state, state_id in state_map.items():
            if aalpy_state.is_accepting:
                container.add_accept_state(State(state_id))

            for symbol_str, target_aalpy_state in aalpy_state.transitions.items():
                try:
                    symbol_id = self.vocab.to_int(symbol_str)
                except KeyError:
                    # RPNILearner と同じ警告処理
                    if hasattr(self.vocab, 'unk_index') and self.vocab.unk_index is not None:
                        symbol_id = self.vocab.unk_index
                    else:
                        print(f"Warning: Symbol '{symbol_str}' from learned DFA not in vocabulary. Skipping transition.")
                        continue
                
                if target_aalpy_state not in state_map:
                    print(f"Warning: Target state {target_aalpy_state.state_id} not in state map. Skipping.")
                    continue

                target_state_id = state_map[target_aalpy_state]

                transition = FiniteAutomatonTransition(
                    state_from=state_id,
                    symbol=symbol_id,
                    state_to=target_state_id,
                    weight=0.0
                )
                container.add_transition(transition)
        
        print(f"Conversion complete. Container has {container.num_states()} states.")
        return container