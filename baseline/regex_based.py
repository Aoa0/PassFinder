import re

# Note: All values here should be lowercase
DENYLIST = (
    'apikey',
    'api_key',
    'aws_secret_access_key',
    'db_pass',
    'password',
    'passwd',
    'private_key',
    'secret',
    'secrete',
)

FALSE_POSITIVES = {
    '""',
    '""):',
    '"\'',
    '")',
    '"dummy',
    '"replace',
    '"this',
    '#pass',
    '#password',
    '$(shell',
    "'\"",
    "''",
    "''):",
    "')",
    "'dummy",
    "'replace",
    "'this",
    '(nsstring',
    '-default}',
    '::',
    '<%=',
    '<?php',
    '<a',
    '<aws_secret_access_key>',
    '<input',
    '<password>',
    '<redacted>',
    '<secret',
    '>',
    '=',
    '\\"$(shell',
    '\\k.*"',
    "\\k.*'",
    '`cat',
    '`grep',
    '`sudo',
    'account_password',
    'api_key',
    'disable',
    'dummy_secret',
    'dummy_value',
    'false',
    'false):',
    'false,',
    'false;',
    'login_password',
    'none',
    'none,',
    'none}',
    'nopasswd',
    'not',
    'not_real_key',
    'null',
    'null,',
    'null.*"',
    "null.*'",
    'null;',
    'pass',
    'pass)',
    'password',
    'password)',
    'password))',
    'password,',
    'password},',
    'prompt',
    'redacted',
    'secret',
    'some_key',
    'str',
    'str_to_sign',
    'string',
    'string)',
    'string,',
    'string;',
    'string?',
    'string?)',
    'string}',
    'string}}',
    'tests',
    'tests-access-key',
    'thisisnottherealsecret',
    'todo',
    'true',
    'true):',
    'true,',
    'true;',
    'undef',
    'undef,',
    '{',
    '{{',
}

# Includes ], ', " as closing
CLOSING = r'[]\'"]{0,2}'
DENYLIST_REGEX = r'|'.join(DENYLIST)
# Non-greedy match
OPTIONAL_WHITESPACE = r'\s*?'
OPTIONAL_NON_WHITESPACE = r'[^\s]*?'
QUOTE = r'[\'"]'
SECRET = r'[^\s]+'
SQUARE_BRACKETS = r'(\[\])'

FOLLOWED_BY_COLON_EQUAL_SIGNS_REGEX = re.compile(
    # e.g. my_password := "bar" or my_password := bar
    r'({denylist})({closing})?{whitespace}:=?{whitespace}({quote}?)({secret})(\3)'.format(
        denylist=DENYLIST_REGEX,
        closing=CLOSING,
        quote=QUOTE,
        whitespace=OPTIONAL_WHITESPACE,
        secret=SECRET,
    ),
)
FOLLOWED_BY_COLON_REGEX = re.compile(
    # e.g. api_key: foo
    r'({denylist})({closing})?:{whitespace}({quote}?)({secret})(\3)'.format(
        denylist=DENYLIST_REGEX,
        closing=CLOSING,
        quote=QUOTE,
        whitespace=OPTIONAL_WHITESPACE,
        secret=SECRET,
    ),
)
FOLLOWED_BY_COLON_QUOTES_REQUIRED_REGEX = re.compile(
    # e.g. api_key: "foo"
    r'({denylist})({closing})?:({whitespace})({quote})({secret})(\4)'.format(
        denylist=DENYLIST_REGEX,
        closing=CLOSING,
        quote=QUOTE,
        whitespace=OPTIONAL_WHITESPACE,
        secret=SECRET,
    ),
)
FOLLOWED_BY_EQUAL_SIGNS_OPTIONAL_BRACKETS_OPTIONAL_AT_SIGN_QUOTES_REQUIRED_REGEX = re.compile(
    # e.g. my_password = "bar"
    # e.g. my_password = @"bar"
    # e.g. my_password[] = "bar";
    r'({denylist})({square_brackets})?{optional_whitespace}={optional_whitespace}(@)?(")({secret})(\5)'.format(
        # noqa: E501
        denylist=DENYLIST_REGEX,
        square_brackets=SQUARE_BRACKETS,
        optional_whitespace=OPTIONAL_WHITESPACE,
        secret=SECRET,
    ),
)
FOLLOWED_BY_EQUAL_SIGNS_REGEX = re.compile(
    # e.g. my_password = bar
    r'({denylist})({closing})?{whitespace}={whitespace}({quote}?)({secret})(\3)'.format(
        denylist=DENYLIST_REGEX,
        closing=CLOSING,
        quote=QUOTE,
        whitespace=OPTIONAL_WHITESPACE,
        secret=SECRET,
    ),
)
FOLLOWED_BY_EQUAL_SIGNS_QUOTES_REQUIRED_REGEX = re.compile(
    # e.g. my_password = "bar"
    r'({denylist})({closing})?{whitespace}={whitespace}({quote})({secret})(\3)'.format(
        denylist=DENYLIST_REGEX,
        closing=CLOSING,
        quote=QUOTE,
        whitespace=OPTIONAL_WHITESPACE,
        secret=SECRET,
    ),
)
FOLLOWED_BY_QUOTES_AND_SEMICOLON_REGEX = re.compile(
    # e.g. private_key "something";
    r'({denylist}){nonWhitespace}{whitespace}({quote})({secret})(\2);'.format(
        denylist=DENYLIST_REGEX,
        nonWhitespace=OPTIONAL_NON_WHITESPACE,
        quote=QUOTE,
        whitespace=OPTIONAL_WHITESPACE,
        secret=SECRET,
    ),
)

DEFAULT_REGEX = [FOLLOWED_BY_COLON_EQUAL_SIGNS_REGEX,
                 FOLLOWED_BY_COLON_REGEX,
                 FOLLOWED_BY_COLON_QUOTES_REQUIRED_REGEX,
                 FOLLOWED_BY_EQUAL_SIGNS_OPTIONAL_BRACKETS_OPTIONAL_AT_SIGN_QUOTES_REQUIRED_REGEX,
                 FOLLOWED_BY_EQUAL_SIGNS_REGEX,
                 FOLLOWED_BY_EQUAL_SIGNS_QUOTES_REQUIRED_REGEX,
                 FOLLOWED_BY_QUOTES_AND_SEMICOLON_REGEX
                 ]

DEFAULT_REGEX_TO_GROUP = {
    FOLLOWED_BY_COLON_EQUAL_SIGNS_REGEX: 4,
    FOLLOWED_BY_COLON_REGEX: 4,
    FOLLOWED_BY_COLON_QUOTES_REQUIRED_REGEX: 5,
    FOLLOWED_BY_EQUAL_SIGNS_OPTIONAL_BRACKETS_OPTIONAL_AT_SIGN_QUOTES_REQUIRED_REGEX: 6,
    FOLLOWED_BY_EQUAL_SIGNS_REGEX: 4,
    FOLLOWED_BY_EQUAL_SIGNS_QUOTES_REQUIRED_REGEX: 5,
    FOLLOWED_BY_QUOTES_AND_SEMICOLON_REGEX: 3,
}


def regex_check(string_lines, custom_regexes=None):
    if custom_regexes is None:
        custom_regexes = {}
    matched_secret = []
    if custom_regexes:
        secret_regexes = custom_regexes
    else:
        secret_regexes = DEFAULT_REGEX_TO_GROUP

    for string in string_lines:
        lowered_string = string.lower()
        for pattern, group_number in secret_regexes.items():
            match = pattern.search(lowered_string)
            if match:
                lowered_secret = match.group(group_number)
                span = match.span(group_number)
                secret = string[span[0]:span[1]]
                if not probably_false_positive(lowered_secret) and heuristic_check(secret):
                    matched_secret.append((match.group(), secret))
                    break

    return matched_secret


def regex_check_line(s, custom_regexes=None):
    if custom_regexes is None:
        custom_regexes = {}
    if custom_regexes:
        secret_regexes = custom_regexes
    else:
        secret_regexes = DEFAULT_REGEX_TO_GROUP

    matched_secret = []
    lowered_string = s.lower()
    for pattern, group_number in secret_regexes.items():
        match = pattern.search(lowered_string)
        if match:
            lowered_secret = match.group(group_number)
            span = match.span(group_number)
            secret = s[span[0]:span[1]]
            if not probably_false_positive(lowered_secret) and heuristic_check(secret):
                matched_secret.append((match.group(), secret))
                break
    return matched_secret


def heuristic_check(lowered_secret):
    # length heuristic
    if len(lowered_secret) < 4 or len(lowered_secret) > 30:
        return False
    return True


def probably_false_positive(lowered_secret):
    if (
            any(
                false_positive in lowered_secret
                for false_positive in (
                        '/etc/',
                        'fake',
                        'forgot',
                )
            ) or lowered_secret in FALSE_POSITIVES
            # For e.g. private_key "some/dir/that/is/not/a/secret";
            or lowered_secret.count('/') >= 3
            # For e.g. "secret": "{secret}"
            or (
            lowered_secret[0] == '{'
            and lowered_secret[-1] == '}'
    )
    ):
        return True

    # Heuristic for no function calls
    try:
        if (
                lowered_secret.index('(') < lowered_secret.index(')')
        ):
            return True
    except ValueError:
        pass

    # Heuristic for e.g. request.json_body['hey']
    try:
        if (
                lowered_secret.index('[') < lowered_secret.index(']')
        ):
            return True
    except ValueError:
        pass

    # Heuristic for e.g. ${link}
    try:
        if (
                lowered_secret.index('${') < lowered_secret.index('}')
        ):
            return True
    except ValueError:
        pass

    return False
