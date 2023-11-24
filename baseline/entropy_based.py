import math
import string

BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
HEX_CHARS = "1234567890abcdefABCDEF"
PRINTABLE_CHARS = string.digits + string.ascii_letters + string.punctuation


def shannon_entropy(data, iterator=PRINTABLE_CHARS):
    """
    Borrowed from http://blog.dkbza.org/2007/05/scanning-data-for-entropy-anomalies.html
    """
    if not data:
        return 0
    entropy = 0
    for x in iterator:
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    # print(entropy)
    return entropy


def get_strings_of_target_set(word, char_set, threshold=5):
    count = 0
    letters = ""
    strings = []
    for char in word:
        if char in char_set:
            letters += char
            count += 1
        else:
            if count > threshold:
                strings.append(letters)
            letters = ""
            count = 0
    if count > threshold:
        strings.append(letters)
    return strings


def check_high_entropy(word):
    high_entropy = False
    b64Entropy = 0
    base64_strings = get_strings_of_target_set(word, BASE64_CHARS)
    hex_strings = get_strings_of_target_set(word, HEX_CHARS)
    # print(base64_strings, hex_strings)
    for string in base64_strings:
        b64Entropy = shannon_entropy(string, BASE64_CHARS)
        if b64Entropy > 3.2:
            high_entropy = True
    # for string in hex_strings:
    #     hexEntropy = shannon_entropy(string, HEX_CHARS)
    #     if (hexEntropy > 3):
    #         high_entropy = True
    # print(b64Entropy, hexEntropy)
    return high_entropy, b64Entropy


def find_high_entropy(string_lines):
    high_entropy_string = []
    lines = string_lines.split("\n")
    for line in lines:
        for word in line.split():
            base64_strings = get_strings_of_target_set(word, BASE64_CHARS)
            hex_strings = get_strings_of_target_set(word, HEX_CHARS)
            for s in base64_strings:
                b64Entropy = shannon_entropy(s, BASE64_CHARS)
                if b64Entropy > 4.5:
                    high_entropy_string.append(s)
            for s in hex_strings:
                hexEntropy = shannon_entropy(s, HEX_CHARS)
                if hexEntropy > 3:
                    high_entropy_string.append(s)
    return high_entropy_string
