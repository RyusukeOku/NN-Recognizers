from rayuela.base.semiring import Real
from rayuela.base.state import State
from rayuela.base.symbol import Sym
from rayuela.fsa.fsa import FSA

from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer
)
from recognizers.hand_picked_languages.rayuela_util import from_rayuela_fsa

def majority_structural_fsa() -> FSA:
    A = FSA(R=Real)
    q_0 = State(0)
    q_1 = State(1)
    A.set_I(q_0, Real.one)
    A.add_arc(q_0, Sym("0"), q_1)
    A.add_arc(q_0, Sym("1"), q_1)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_1, s, q_1)
    A.set_F(q_1)
    return A

def unmarked_reversal_structural_fsa() -> FSA:
    return majority_structural_fsa()

def stack_manipulation_structural_fsa() -> FSA:
    A = FSA(R=Real)
    q_0 = State(0)
    q_1 = State(1)
    q_2 = State(2)
    q_3 = State(3)

    A.set_I(q_0, Real.one)
    A.add_arc(q_0, Sym("0"), q_1)
    A.add_arc(q_0, Sym("1"), q_1)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_1, s, q_1)
    A.add_arc(q_1, Sym("POP"), q_2)
    A.add_arc(q_2, Sym("PUSH"), q_2)
    for s in [Sym("POP"), Sym("PUSH"), Sym(" ")]:
        A.add_arc(q_2, s, q_2)
    A.add_arc(q_2, Sym("#"), q_3)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_3, s, q_3)
    A.set_F(q_3)
    return A

def marked_reversal_structural_fsa() -> FSA:
    A = FSA(R=Real)
    q_0 = State(0)
    q_1 = State(1)
    q_2 = State(2)

    A.set_I(q_0, Real.one)
    A.add_arc(q_0, Sym("0"), q_1)
    A.add_arc(q_0, Sym("1"), q_1)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_1, s, q_1)
    A.add_arc(q_1, Sym("#"), q_2)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_2, s, q_2)
    A.set_F(q_2)
    return A

def marked_copy_structural_fsa() -> FSA:
    return marked_reversal_structural_fsa()

def missing_duplicate_structural_fsa() -> FSA:
    A = FSA(R=Real)
    q_0 = State(0)
    q_1 = State(1)
    q_2 = State(2)

    A.set_I(q_0, Real.one)
    A.add_arc(q_0, Sym("0"), q_1)
    A.add_arc(q_0, Sym("1"), q_1)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_1, s, q_1)
    A.add_arc(q_1, Sym("_"), q_2)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_2, s, q_2)
    A.set_F(q_2)
    return A

def odds_first_structural_fsa() -> FSA:
    A = FSA(R=Real)
    q_0 = State(0)
    q_1 = State(1)
    q_2 = State(2)

    A.set_I(q_0, Real.one)
    A.add_arc(q_0, Sym("0"), q_1)
    A.add_arc(q_0, Sym("1"), q_1)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_1, s, q_1)
    A.add_arc(q_1, Sym("="), q_2)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_2, s, q_2)
    A.set_F(q_2)
    return A

def compute_sqrt_structural_fsa() -> FSA:
    return odds_first_structural_fsa()

def binary_addition_structural_fsa() -> FSA:
    A = FSA(R=Real)
    q_0 = State(0)
    q_1 = State(1)
    q_2 = State(2)
    q_3 = State(3)

    A.set_I(q_0, Real.one)
    A.add_arc(q_0, Sym("0"), q_1)
    A.add_arc(q_0, Sym("1"), q_1)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_1, s, q_1)
    A.add_arc(q_1, Sym("+"), q_2)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_2, s, q_2)
    A.add_arc(q_2, Sym("="), q_3)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_3, s, q_3)
    A.set_F(q_3)
    return A

def binary_multiplication_structural_fsa() -> FSA:
    A = FSA(R=Real)
    q_0 = State(0)
    q_1 = State(1)
    q_2 = State(2)
    q_3 = State(3)

    A.set_I(q_0, Real.one)
    A.add_arc(q_0, Sym("0"), q_1)
    A.add_arc(q_0, Sym("1"), q_1)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_1, s, q_1)
    A.add_arc(q_1, Sym("×"), q_2)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_2, s, q_2)
    A.add_arc(q_2, Sym("="), q_3)
    for s in [Sym("0"), Sym("1"), Sym(" ")]:
        A.add_arc(q_3, s, q_3)
    A.set_F(q_3)
    return A

def bucket_sort_structural_fsa() -> FSA:
    A = FSA(R=Real)
    q_0 = State(0)
    q_1 = State(1)
    q_2 = State(2)

    A.set_I(q_0, Real.one)
    A.add_arc(q_0, Sym("1"), q_1)
    A.add_arc(q_0, Sym("2"), q_1)
    A.add_arc(q_0, Sym("3"), q_1)
    A.add_arc(q_0, Sym("4"), q_1)
    A.add_arc(q_0, Sym("5"), q_1)
    for s in [Sym("1"), Sym("2"), Sym("3"), Sym("4"), Sym("5"), Sym(" ")]:
        A.add_arc(q_1, s, q_1)
    A.add_arc(q_1, Sym("#"), q_2)
    for s in [Sym("1"), Sym("2"), Sym("3"), Sym("4"), Sym("5"), Sym(" ")]:
        A.add_arc(q_2, s, q_2)
    A.set_F(q_2)
    return A


def majority_structural_fsa_container() -> FiniteAutomatonContainer:
    return from_rayuela_fsa(majority_structural_fsa())

def unmarked_reversal_structural_fsa_container() -> FiniteAutomatonContainer:
    return from_rayuela_fsa(unmarked_reversal_structural_fsa())

def stack_manipulation_structural_fsa_container() -> FiniteAutomatonContainer:
    return from_rayuela_fsa(stack_manipulation_structural_fsa())

def marked_reversal_structural_fsa_container() -> FiniteAutomatonContainer:
    return from_rayuela_fsa(marked_reversal_structural_fsa())

def marked_copy_structural_fsa_container() -> FiniteAutomatonContainer:
    return from_rayuela_fsa(marked_copy_structural_fsa())

def missing_duplicate_structural_fsa_container() -> FiniteAutomatonContainer:
    return from_rayuela_fsa(missing_duplicate_structural_fsa())

def odds_first_structural_fsa_container() -> FiniteAutomatonContainer:
    return from_rayuela_fsa(odds_first_structural_fsa())

def compute_sqrt_structural_fsa_container() -> FiniteAutomatonContainer:
    return from_rayuela_fsa(compute_sqrt_structural_fsa())

def binary_addition_structural_fsa_container() -> FiniteAutomatonContainer:
    return from_rayuela_fsa(binary_addition_structural_fsa())

def binary_multiplication_structural_fsa_container() -> FiniteAutomatonContainer:
    return from_rayuela_fsa(binary_multiplication_structural_fsa())

def bucket_sort_structural_fsa_container() -> FiniteAutomatonContainer:
    return from_rayuela_fsa(bucket_sort_structural_fsa())