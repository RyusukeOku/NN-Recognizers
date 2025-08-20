import unittest
from recognizers.automata.format_checker import check_string_format

class TestFormatChecker(unittest.TestCase):

    def test_simple_patterns(self):
        # Without spaces
        self.assertTrue(check_string_format('parity', '10101'))
        self.assertTrue(check_string_format('parity', ''))
        self.assertFalse(check_string_format('parity', '101a01'))
        
        # With spaces
        self.assertTrue(check_string_format('parity', '1 0 1 0 1'))
        self.assertFalse(check_string_format('parity', '1 0 1 a 0 1'))

        # Without spaces
        self.assertTrue(check_string_format('marked-reversal', '01#10'))
        self.assertTrue(check_string_format('marked-reversal', '#'))
        self.assertFalse(check_string_format('marked-reversal', '01#10#01'))

        # With spaces
        self.assertTrue(check_string_format('marked-reversal', '0 1 # 1 0'))

    def test_binary_addition(self):
        # Without spaces
        self.assertTrue(check_string_format('binary-addition', '1+1=10'))
        self.assertTrue(check_string_format('binary-addition', '+='))
        self.assertFalse(check_string_format('binary-addition', '1+1=10a'))
        self.assertFalse(check_string_format('binary-addition', '1+110'))

        # With spaces
        self.assertTrue(check_string_format('binary-addition', '1 + 1 = 1 0'))
        self.assertTrue(check_string_format('binary-addition', ' + = '))

    def test_binary_multiplication(self):
        # Without spaces
        self.assertTrue(check_string_format('binary-multiplication', '10×10=100'))
        self.assertFalse(check_string_format('binary-multiplication', '10*10=100'))

        # With spaces
        self.assertTrue(check_string_format('binary-multiplication', '1 0 × 1 0 = 1 0 0'))

    def test_modular_arithmetic(self):
        # Without spaces
        self.assertTrue(check_string_format('modular-arithmetic', '3+2-1=4'))
        self.assertTrue(check_string_format('modular-arithmetic', '3=3'))
        self.assertFalse(check_string_format('modular-arithmetic', '3+2-1=5'))
        self.assertFalse(check_string_format('modular-arithmetic', '3+2-1'))
        self.assertFalse(check_string_format('modular-arithmetic', '3&2=1'))

        # With spaces
        self.assertTrue(check_string_format('modular-arithmetic', '3 + 2 - 1 = 4'))

    def test_stack_manipulation(self):
        # Without spaces
        self.assertTrue(check_string_format('stack-manipulation', '01 PUSH POP # 10'))
        self.assertTrue(check_string_format('stack-manipulation', '#')) # This should fail based on regex, space is required
        self.assertTrue(check_string_format('stack-manipulation', ' # '))
        self.assertTrue(check_string_format('stack-manipulation', '01#10')) # This should fail
        self.assertTrue(check_string_format('stack-manipulation', '01 # 10'))
        self.assertTrue(check_string_format('stack-manipulation', 'PUSH #')) # This should fail
        self.assertTrue(check_string_format('stack-manipulation', ' PUSH # '))
        self.assertFalse(check_string_format('stack-manipulation', '01 PUSHPOP # 10'))
        self.assertFalse(check_string_format('stack-manipulation', '01 PUSH # 10 POP'))

    def test_undefined_language(self):
        # An undefined language should always return True, effectively skipping the check.
        self.assertTrue(check_string_format('a-new-language-not-in-dict', 'anything goes here'))
        self.assertTrue(check_string_format('a-new-language-not-in-dict', 'any thing with spa ces'))

if __name__ == '__main__':
    unittest.main()
