
from rayuela.base.semiring import Boolean
from rayuela.base.state import State
from rayuela.base.symbol import Sym
from rayuela.fsa.fsa import FSA

from recognizers.automata.finite_automaton import FiniteAutomatonContainer
from recognizers.hand_picked_languages.rayuela_util import from_rayuela_fsa

def majority_structural_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    """ Matches [01]^* """
    alphabet = ['0', '1']
    fsa = FSA(R=Boolean)
    q0 = State(0)
    fsa.set_I(q0, Boolean.one)
    fsa.set_F(q0, Boolean.one)
    fsa.add_arc(q0, Sym('0'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('1'), q0, Boolean.one)
    return from_rayuela_fsa(fsa, alphabet)

def reversal_structural_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    """ Matches [01]^* """
    return majority_structural_dfa()

def marked_reversal_structural_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    """ Matches [01]^* # [01]^* """
    alphabet = ['0', '1', '#']
    fsa = FSA(R=Boolean)
    q0, q1, q2 = State(0), State(1), State(2)
    fsa.set_I(q0, Boolean.one)
    fsa.add_arc(q0, Sym('0'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('1'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('#'), q1, Boolean.one)
    fsa.add_arc(q1, Sym('0'), q2, Boolean.one)
    fsa.add_arc(q1, Sym('1'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('0'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('1'), q2, Boolean.one)
    fsa.set_F(q2, Boolean.one)
    return from_rayuela_fsa(fsa, alphabet)

def marked_copy_structural_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    """ Matches [01]^* # [01]^* """
    return marked_reversal_structural_dfa()

def stack_manipulation_structural_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    """ Matches [01]^* [POP|PUSH}^* # [01]^* """
    alphabet = ['0', '1', 'POP', 'PUSH', '#']
    fsa = FSA(R=Boolean)
    q0, q1, q2, q3 = State(0), State(1), State(2), State(3)
    fsa.set_I(q0, Boolean.one)
    fsa.add_arc(q0, Sym('0'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('1'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('POP'), q1, Boolean.one)
    fsa.add_arc(q0, Sym('PUSH'), q1, Boolean.one)
    fsa.add_arc(q1, Sym('POP'), q1, Boolean.one)
    fsa.add_arc(q1, Sym('PUSH'), q1, Boolean.one)
    fsa.add_arc(q1, Sym('#'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('0'), q3, Boolean.one)
    fsa.add_arc(q2, Sym('1'), q3, Boolean.one)
    fsa.add_arc(q3, Sym('0'), q3, Boolean.one)
    fsa.add_arc(q3, Sym('1'), q3, Boolean.one)
    fsa.set_F(q3, Boolean.one)
    return from_rayuela_fsa(fsa, alphabet)

def missing_duplicate_structural_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    """ Matches [01]^* _ [01]^* """
    alphabet = ['0', '1', '_']
    fsa = FSA(R=Boolean)
    q0, q1, q2 = State(0), State(1), State(2)
    fsa.set_I(q0, Boolean.one)
    fsa.add_arc(q0, Sym('0'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('1'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('_'), q1, Boolean.one)
    fsa.add_arc(q1, Sym('0'), q2, Boolean.one)
    fsa.add_arc(q1, Sym('1'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('0'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('1'), q2, Boolean.one)
    fsa.set_F(q2, Boolean.one)
    return from_rayuela_fsa(fsa, alphabet)

def odds_first_structural_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    """ Matches [01]^* = [01]^* """
    alphabet = ['0', '1', '=']
    fsa = FSA(R=Boolean)
    q0, q1, q2 = State(0), State(1), State(2)
    fsa.set_I(q0, Boolean.one)
    fsa.add_arc(q0, Sym('0'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('1'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('='), q1, Boolean.one)
    fsa.add_arc(q1, Sym('0'), q2, Boolean.one)
    fsa.add_arc(q1, Sym('1'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('0'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('1'), q2, Boolean.one)
    fsa.set_F(q2, Boolean.one)
    return from_rayuela_fsa(fsa, alphabet)

def compute_sqrt_structural_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    """ Matches [01]^* = [01]^* """
    return odds_first_structural_dfa()

def binary_addition_structural_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    """ Matches [01]^* + [01]^* = [01]^* """
    alphabet = ['0', '1', '+', '=']
    fsa = FSA(R=Boolean)
    q0, q1, q2, q3, q4 = State(0), State(1), State(2), State(3), State(4)
    fsa.set_I(q0, Boolean.one)
    fsa.add_arc(q0, Sym('0'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('1'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('+'), q1, Boolean.one)
    fsa.add_arc(q1, Sym('0'), q2, Boolean.one)
    fsa.add_arc(q1, Sym('1'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('0'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('1'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('='), q3, Boolean.one)
    fsa.add_arc(q3, Sym('0'), q4, Boolean.one)
    fsa.add_arc(q3, Sym('1'), q4, Boolean.one)
    fsa.add_arc(q4, Sym('0'), q4, Boolean.one)
    fsa.add_arc(q4, Sym('1'), q4, Boolean.one)
    fsa.set_F(q4, Boolean.one)
    return from_rayuela_fsa(fsa, alphabet)

def binary_multiplication_structural_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    """ Matches [01]^* × [01]^* = [01]^* """
    alphabet = ['0', '1', '×', '=']
    fsa = FSA(R=Boolean)
    q0, q1, q2, q3, q4 = State(0), State(1), State(2), State(3), State(4)
    fsa.set_I(q0, Boolean.one)
    fsa.add_arc(q0, Sym('0'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('1'), q0, Boolean.one)
    fsa.add_arc(q0, Sym('×'), q1, Boolean.one)
    fsa.add_arc(q1, Sym('0'), q2, Boolean.one)
    fsa.add_arc(q1, Sym('1'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('0'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('1'), q2, Boolean.one)
    fsa.add_arc(q2, Sym('='), q3, Boolean.one)
    fsa.add_arc(q3, Sym('0'), q4, Boolean.one)
    fsa.add_arc(q3, Sym('1'), q4, Boolean.one)
    fsa.add_arc(q4, Sym('0'), q4, Boolean.one)
    fsa.add_arc(q4, Sym('1'), q4, Boolean.one)
    fsa.set_F(q4, Boolean.one)
    return from_rayuela_fsa(fsa, alphabet)

def bucket_sort_structural_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    """ Matches [1-5]^* # [1-5]^* """
    alphabet = ['1', '2', '3', '4', '5', '#']
    fsa = FSA(R=Boolean)
    q0, q1, q2 = State(0), State(1), State(2)
    fsa.set_I(q0, Boolean.one)
    for i in range(1, 6):
        fsa.add_arc(q0, Sym(str(i)), q0, Boolean.one)
        fsa.add_arc(q1, Sym(str(i)), q2, Boolean.one)
        fsa.add_arc(q2, Sym(str(i)), q2, Boolean.one)
    fsa.add_arc(q0, Sym('#'), q1, Boolean.one)
    fsa.set_F(q2, Boolean.one)
    return from_rayuela_fsa(fsa, alphabet)
