from collections.abc import Iterable
from typing import TypeVar

from .automaton import (
    State,
    Transition,
    Automaton,
    AutomatonContainer,
    WeightedAutomaton,
    WeightedAutomatonContainer
)
from .semiring import Semiring

from rayuela.fsa.fsa import FSA
from rayuela.base.semiring import Tropical

Weight = TypeVar('Weight')

class FiniteAutomatonTransition(Transition):
    pass

class FiniteAutomaton(Automaton):

    def transitions(self) -> Iterable[FiniteAutomatonTransition]:
        return super().transitions() # type: ignore

class FiniteAutomatonContainer(FiniteAutomaton, AutomatonContainer):

    _accept_states: dict[State, None]

    def __init__(self,
        *,
        num_states: int=1,
        alphabet_size: int,
        initial_state: State=State(0)
    ):
        super().__init__(
            num_states=num_states,
            alphabet_size=alphabet_size,
            initial_state=initial_state
        )
        self._accept_states = {}

    def add_transition(self, transition: FiniteAutomatonTransition) -> None:
        self._add_transition(transition)

    def is_accept_state(self, state: State) -> bool:
        return state in self._accept_states

    def add_accept_state(self, state: State) -> None:
        self._accept_states[state] = None

    def to_rayuela_fsa(self, alphabet: list[str]) -> FSA:
        """
        Converts this automaton to a Rayuela FSA object.
        Args:
            alphabet: A list of strings representing the alphabet, where the
                      index corresponds to the symbol ID.
        Returns:
            A Rayuela FSA object.
        """
        fsa = FSA(R=Tropical)
        for i in range(self.num_states()):
            fsa.add_state(i)
        fsa.set_I(self.initial_state())
        for state in range(self.num_states()):
            if self.is_accept_state(state):
                fsa.add_F(state, Tropical(0.0))
        for transition in self.transitions():
            source_state = transition.state_from
            label_index = transition.symbol
            next_state = transition.state_to
            token = alphabet[label_index]
            fsa.add_arc(source_state, token, next_state, Tropical(0.0))
        return fsa

class WeightedFiniteAutomaton(FiniteAutomaton, WeightedAutomaton[Weight]):

    def transition_weights(self) -> Iterable[tuple[FiniteAutomatonTransition, Weight]]:
        return super().transition_weights() # type: ignore

    def accept_weights(self) -> Iterable[tuple[State, Weight]]:
        raise NotImplementedError

class WeightedFiniteAutomatonContainer(WeightedFiniteAutomaton[Weight], WeightedAutomatonContainer[Weight]):

    _accept_weights: dict[State, Weight]

    def __init__(self,
        *,
        num_states: int=1,
        alphabet_size: int,
        initial_state: State=State(0),
        semiring: Semiring[Weight]
    ):
        super().__init__(
            num_states=num_states,
            alphabet_size=alphabet_size,
            initial_state=initial_state,
            semiring=semiring
        )
        self._accept_weights = {}

    def set_transition_weight(self,
        transition: FiniteAutomatonTransition,
        weight: Weight
    ) -> None:
        self._set_transition_weight(transition, weight)

    def is_accept_state(self, state: State) -> bool:
        return state in self._accept_weights

    def accept_weights(self) -> Iterable[tuple[State, Weight]]:
        return self._accept_weights.items()

    def set_accept_weight(self, state: State, weight: Weight) -> None:
        self._accept_weights[state] = weight

class FiniteAutomatonRunner:
    """A class to run a FiniteAutomaton on input sequences."""

    def __init__(self, automaton: FiniteAutomaton):
        self.automaton = automaton
        self._transitions = {}
        for transition in self.automaton.transitions():
            key = (transition.state_from, transition.symbol)
            self._transitions[key] = transition.state_to

    def get_state_sequence(self, input_symbols: list[str], alphabet: dict[str, int]) -> list[int]:
        """ 
        Processes an input string and returns the sequence of states.
        The initial state is included as the first element.
        """
        current_state = self.automaton.initial_state()
        # The first state, before consuming any symbols, is the initial state.
        states = [current_state]

        for symbol_str in input_symbols:
            if symbol_str not in alphabet:
                # If the symbol is not in the FSA's alphabet, it's a failure.
                # We can handle this by transitioning to a non-accepting sink state
                # if one exists, or simply staying put. For this use case,
                # we assume the input vocabulary is aligned with the FSA's alphabet.
                # We will stay in the current state if symbol is unknown.
                pass
            else:
                symbol = alphabet[symbol_str]
                current_state = self._transitions.get(
                    (current_state, symbol), 
                    current_state # Default to staying in the same state if transition not defined
                )
            states.append(current_state)
        
        return states
