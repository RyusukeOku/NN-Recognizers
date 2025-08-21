
from recognizers.automata.finite_automaton import FiniteAutomatonContainer
from recognizers.hand_picked_languages import (
    parity, even_pairs, repeat_01, cycle_navigation, modular_arithmetic_simple, dyck_k_m, first
)
from recognizers.automata.structural_fsas import (
    majority_structural_dfa, reversal_structural_dfa, stack_manipulation_structural_dfa,
    marked_reversal_structural_dfa, marked_copy_structural_dfa,
    missing_duplicate_structural_dfa, odds_first_structural_dfa,
    compute_sqrt_structural_dfa, binary_addition_structural_dfa,
    binary_multiplication_structural_dfa, bucket_sort_structural_dfa
)

FSA_MAP = {
    'parity': parity.parity_dfa,
    'even-pairs': even_pairs.even_pairs_dfa,
    'repeat-01': repeat_01.repeat_01_dfa,
    'cycle-navigation': cycle_navigation.cycle_navigation_dfa,
    'modular-arithmetic': modular_arithmetic_simple.modular_arithmetic_dfa,
    'dyck-2-3': lambda: dyck_k_m.dyck_k_m_dfa(k=2, m=3),
    'first': first.first_dfa,
    'majority_structural': majority_structural_dfa,
    'reversal_structural': reversal_structural_dfa,
    'stack-manipulation_structural': stack_manipulation_structural_dfa,
    'marked-reversal_structural': marked_reversal_structural_dfa,
    'marked-copy_structural': marked_copy_structural_dfa,
    'missing-duplicate_structural': missing_duplicate_structural_dfa,
    'odds-first_structural': odds_first_structural_dfa,
    'compute-sqrt_structural': compute_sqrt_structural_dfa,
    'binary-addition_structural': binary_addition_structural_dfa,
    'binary-multiplication_structural': binary_multiplication_structural_dfa,
    'bucket-sort_structural': bucket_sort_structural_dfa,
}

# For some languages, the FSA to integrate is the one that checks the structure,
# not the one that validates the language itself.
STRUCTURAL_FSA_LANGUAGES = [
    'majority', 'unmarked-reversal', 'stack-manipulation', 'marked-reversal',
    'marked-copy', 'missing-duplicate', 'odds-first', 'compute-sqrt',
    'binary-addition', 'binary-multiplication', 'bucket-sort'
]

def get_fsa_name_for_language(language: str) -> str:
    if language in STRUCTURAL_FSA_LANGUAGES:
        # e.g. 'binary_addition' -> 'binary_addition_structural'
        # e.g. 'unmarked_reversal' -> 'reversal_structural'
        if language == 'unmarked-reversal':
            return 'reversal_structural'
        return f'{language}_structural'
    else:
        # For languages like 'parity', the language name is the fsa name.
        # Also handles 'dyck_k_m' which needs special construction.
        if language == 'dyck_k_m':
            return 'dyck_2_3'
        return language

def get_fsa(name: str) -> tuple[FiniteAutomatonContainer, list[str]]:
    if name not in FSA_MAP:
        raise ValueError(f"Unknown FSA name: {name}")
    return FSA_MAP[name]()
