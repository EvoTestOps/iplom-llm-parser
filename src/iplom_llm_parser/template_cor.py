import regex as re


def correct_single_template(template):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean)
    US (User String)
    DG (Digit)
    PS (Path-like String)
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """

    path_delimiters = {
        r"\s",
        r"\,",
        r"\!",
        r"\;",
        r"\:",
        r"\=",
        r"\|",
        r"\"",
        r"\'",
        r"\[",
        r"\]",
        r"\(",
        r"\)",
        r"\{",
        r"\}",
    }
    token_delimiters = path_delimiters.union(
        {
            r"\.",
            r"\-",
            r"\+",
            r"\@",
            r"\#",
            r"\$",
            r"\%",
            r"\&",
        }
    )

    # DS
    template = template.strip()
    template = re.sub(r"\s+", " ", template)

    # PS
    p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
    new_p_tokens = []
    for p_token in p_tokens:
        if re.match(r'^(\/[^\/]+)+$', p_token):
            p_token = '<*>'
        new_p_tokens.append(p_token)
    template = ''.join(new_p_tokens)

    tokens = re.split("(" + "|".join(token_delimiters) + ")", template)
    new_tokens = []
    for token in tokens:
        # DG
        if re.match(r"^\d+$", token):
            token = "<*>"

        # WV
        if re.match(r"^[^\s\/]*<\*>[^\s\/]*$", token):
            if token != "<*>/<*>":
                token = "<*>"

        new_tokens.append(token)

    template = "".join(new_tokens)

    # DV
    while True:
        prev = template
        template = re.sub(r"<\*>\.<\*>", "<*>", template)
        if prev == template:
            break

    # CV
    while True:
        prev = template
        template = re.sub(r"<\*><\*>", "<*>", template)
        if prev == template:
            break

    while " #<*># " in template:
        template = template.replace(" #<*># ", " <*> ")

    while " #<*> " in template:
        template = template.replace(" #<*> ", " <*> ")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>#<*>" in template:
        template = template.replace("<*>#<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while "<*>@<*>" in template:
        template = template.replace("<*>@<*>", "<*>")

    while "<*>.<*>" in template:
        template = template.replace("<*>.<*>", "<*>")

    while ' "<*>" ' in template:
        template = template.replace(' "<*>" ', " <*> ")

    while " '<*>' " in template:
        template = template.replace(" '<*>' ", " <*> ")

    while "<*><*>" in template:
        template = template.replace("<*><*>", "<*>")

    # List with space separator (>=5 placeholders).
    template = re.sub(r"<\*>(?: <\*>){4,}", "<*>", template)

    return template
