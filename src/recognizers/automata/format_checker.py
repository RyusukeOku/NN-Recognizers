import re

# Note on regex patterns:
# - `stack-manipulation`: The pattern does not contain spaces. The checking function
#   will split the input string by spaces and join them before matching.
# - `modular-arithmetic`: The `-` in `[+\-x]` is escaped to be treated as a literal.
FORMAT_PATTERNS = {
    'even-pairs': r'^[01]*$',
    'repeat-01': r'^[01]*$',
    'parity': r'^[01]*$',
    'first': r'^[01]*$',
    'majority': r'^[01]*$',
    'unmarked-reversal': r'^[01]*$',
    'cycle-navigation': r'^[<>=]*[0-4]$',
    'stack-manipulation': r'^[01]*((POP)|(PUSH))*#[01]*$',
    'marked-reversal': r'^[01]*#[01]*$',
    'marked-copy': r'^[01]*#[01]*$',
    'missing-duplicate': r'^[01]*_[01]*$',
    'odds-first': r'^[01]*=[01]*$',
    'compute-sqrt': r'^[01]*=[01]*$',
    'binary-addition': r'^[01]*\+[01]*=[01]*$',
    'binary-multiplication': r'^[01]*×[01]*=[01]*$',
    'bucket-sort': r'^[1-5]*#[1-5]*$',
    'modular-arithmetic': r'^[0-4]([+\-x][0-4])*=[0-4]$',
}

def check_string_format(language_name: str, input_string: str) -> bool:
    """
    Checks if the input string matches the format for the given language.
    Returns True if the format is valid or if no pattern is defined for the language.
    """
    pattern = FORMAT_PATTERNS.get(language_name)
    if pattern is None:
        return True
    
    # Split the string into tokens and join them to handle tokenized inputs
    # while preserving multi-character tokens like "PUSH".
    tokens = input_string.split(' ')
    processed_string = "".join(tokens)

    return re.fullmatch(pattern, processed_string) is not None