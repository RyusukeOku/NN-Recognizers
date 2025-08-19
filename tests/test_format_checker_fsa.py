import unittest

from recognizers.automata.format_checker_fsa import get_format_checker_fsa

class TestFormatCheckerFSA(unittest.TestCase):

    def test_binary_addition_fsa(self):
        fsa = get_format_checker_fsa('binary-addition')
        self.assertIsNotNone(fsa)

        # Should be accepted
        self.assertTrue(fsa.accept('101+11=1000') == fsa.R.one)
        self.assertTrue(fsa.accept('0+0=0') == fsa.R.one)
        self.assertTrue(fsa.accept('+') == fsa.R.zero) # must have equals
        self.assertTrue(fsa.accept('=') == fsa.R.zero) # must have plus
        self.assertTrue(fsa.accept('1+1=') == fsa.R.one)
        self.assertTrue(fsa.accept('+1=1') == fsa.R.one)
        self.assertTrue(fsa.accept('1+=1') == fsa.R.one)

        # Should be rejected
        self.assertTrue(fsa.accept('101+11') == fsa.R.zero)
        self.assertTrue(fsa.accept('101=1000') == fsa.R.zero)
        self.assertTrue(fsa.accept('abc+def=ghi') == fsa.R.zero)
        self.assertTrue(fsa.accept('1+1=1+1') == fsa.R.zero)
        self.assertTrue(fsa.accept('1+1=1=1') == fsa.R.zero)
        self.assertTrue(fsa.accept('1+1+1=1') == fsa.R.zero)
        self.assertTrue(fsa.accept('a+1=1') == fsa.R.zero)

    def test_parity_fsa(self):
        fsa = get_format_checker_fsa('parity')
        self.assertIsNotNone(fsa)

        # Should be accepted
        self.assertTrue(fsa.accept('10101') == fsa.R.one)
        self.assertTrue(fsa.accept('000') == fsa.R.one)
        self.assertTrue(fsa.accept('') == fsa.R.one)

        # Should be rejected
        self.assertTrue(fsa.accept('abc') == fsa.R.zero)
        self.assertTrue(fsa.accept('101201') == fsa.R.zero)

    def test_undefined_language(self):
        fsa = get_format_checker_fsa('some-undefined-language')
        self.assertIsNone(fsa)

if __name__ == '__main__':
    unittest.main()
