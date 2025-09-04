import unittest

from rayuela.base.symbol import Sym

from recognizers.automata.structural_fsas import (
    majority_structural_fsa,
    stack_manipulation_structural_fsa,
    marked_reversal_structural_fsa,
    missing_duplicate_structural_fsa,
    odds_first_structural_fsa,
    binary_addition_structural_fsa,
    binary_multiplication_structural_fsa,
    bucket_sort_structural_fsa,
)


class TestStructuralFSAs(unittest.TestCase):
    """
    Tests for the FSAs defined in structural_fsas.py.
    It checks whether each FSA correctly accepts or rejects given strings.
    """

    def _check_acceptance(self, fsa, string: str, should_accept: bool):
        """
        Helper function to check if an FSA accepts a string.
        It computes the pathsum for the string and checks if it's non-zero.
        """
        string_as_syms = [Sym(c) for c in string]
        weight = fsa.pathsum(string_as_syms)

        if should_accept:
            self.assertGreater(
                weight.value, 0.0,
                f"FSA should have accepted '{string}' but did not."
            )
        else:
            self.assertEqual(
                weight.value, 0.0,
                f"FSA should have rejected '{string}' but did not."
            )

    def _run_test_cases(self, fsa, accept_cases, reject_cases):
        """Runs a set of accept and reject test cases for a given FSA."""
        for s in accept_cases:
            with self.subTest(string=s, expected="accept"):
                self._check_acceptance(fsa, s, True)

        for s in reject_cases:
            with self.subTest(string=s, expected="reject"):
                self._check_acceptance(fsa, s, False)

    def test_majority_fsa(self):
        fsa = majority_structural_fsa()
        accept = ["0", "1", "01", "1 0", "000"]
        reject = [""]
        self._run_test_cases(fsa, accept, reject)

    def test_stack_manipulation_fsa(self):
        fsa = stack_manipulation_structural_fsa()
        accept = ["1 0 POP # 0", "0 1 PUSH # 1", "0 PUSH # 1"]
        reject = ["#", "POP", "PUSH", "01", ""]
        self._run_test_cases(fsa, accept, reject)

    def test_marked_reversal_fsa(self):
        fsa = marked_reversal_structural_fsa()
        accept = ["0 1 # 1 0", "0#1", "10#11"]
        reject = ["01", "#", ""]
        self._run_test_cases(fsa, accept, reject)

    def test_missing_duplicate_fsa(self):
        fsa = missing_duplicate_structural_fsa()
        accept = ["_", "0_1", " _ "]
        reject = ["01", "_", ""]
        self._run_test_cases(fsa, accept, reject)

    def test_odds_first_fsa(self):
        fsa = odds_first_structural_fsa()
        accept = ["=", "1=1", " = "]
        reject = ["11", " ", ""]
        self._run_test_cases(fsa, accept, reject)

    def test_binary_addition_fsa(self):
        fsa = binary_addition_structural_fsa()
        accept = ["+=", "0+1=1", "1 + 1 = 10"]
        reject = ["+", "=", "1=1+0", ""]
        self._run_test_cases(fsa, accept, reject)

    def test_binary_multiplication_fsa(self):
        fsa = binary_multiplication_structural_fsa()
        accept = ["×=", "1×1=1", " 1 × 0 = 0 "]
        reject = ["×", "=", "1=1×0", ""]
        self._run_test_cases(fsa, accept, reject)

    def test_bucket_sort_fsa(self):
        fsa = bucket_sort_structural_fsa()
        accept = ["#", "123#321", " 5#1 "]
        reject = ["123", " ", ""]
        self._run_test_cases(fsa, accept, reject)


if __name__ == '__main__':
    unittest.main()
