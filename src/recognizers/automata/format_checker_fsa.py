from rayuela.base.semiring import Boolean
from rayuela.base.state import State
from rayuela.base.symbol import Sym
from rayuela.fsa.fsa import FSA

def get_binary_addition_fsa() -> FSA:
    """
    Constructs an FSA for the regular expression `[0-1]*+[0-1]*=[0-1]*`.
    """
    fsa = FSA(R=Boolean)
    q0, q1, q2 = State(0), State(1), State(2)

    fsa.add_arc(q0, Sym('0'), q0)
    fsa.add_arc(q0, Sym('1'), q0)
    fsa.add_arc(q0, Sym('+'), q1)

    fsa.add_arc(q1, Sym('0'), q1)
    fsa.add_arc(q1, Sym('1'), q1)
    fsa.add_arc(q1, Sym('='), q2)

    fsa.add_arc(q2, Sym('0'), q2)
    fsa.add_arc(q2, Sym('1'), q2)

    fsa.set_I(q0, Boolean.one)
    fsa.set_F(q2, Boolean.one)
    
    return fsa

def get_parity_fsa() -> FSA:
    """
    Constructs an FSA for the regular expression `[01]*`.
    """
    fsa = FSA(R=Boolean)
    q0 = State(0)
    fsa.add_arc(q0, Sym('0'), q0)
    fsa.add_arc(q0, Sym('1'), q0)
    fsa.set_I(q0, Boolean.one)
    fsa.set_F(q0, Boolean.one)
    return fsa

def get_format_checker_fsa(language_name: str) -> FSA | None:
    """
    Returns a format-checking FSA for the given language.
    Returns None if a specific format checker is not defined for the language.
    """
    if language_name == 'binary-addition':
        return get_binary_addition_fsa()
    if language_name == 'parity':
        return get_parity_fsa()
    
    # Add other languages here
    # e.g.
    # if language_name == 'dyck-2-3':
    #     return get_dyck_fsa()

    return None
