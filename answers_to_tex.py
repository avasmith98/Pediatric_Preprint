import functools
import itertools
import json
import re

ANSWER_REGEXP = re.compile(r"<answer>(.*)</answer>", re.DOTALL)
EXPLANATION_REGEXP = re.compile(r"<explain>(.*)</explain>", re.DOTALL)
REFERENCE_BOUNDARY_REGEXP = re.compile(r"(\. \[.*?\]) ([A-Z])")
PMID_REGEXP = re.compile(r"PMID: (\d+)")

PREAMBLE = [
    r"\documentclass{article}",
    r"\usepackage[letterpaper, portrait, margin=1in]{geometry}",
    r"\usepackage{hyperref}"
    r"\usepackage[utf8]{inputenc}",
    r"\hypersetup{colorlinks=true}",
    r"\title{Drug Safety Output}",
    r"\begin{document}",
    r"\maketitle",
]
POSTAMBLE = [
    r"\end{document}",
]


def escape_latex_chars(text: str) -> str:
    latex_chars = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    regex = re.compile(
        "|".join([re.escape(char) for char in latex_chars.keys()])
    )
    escaped = re.sub(regex, lambda m: latex_chars[m.group(0)], text)
    return escaped


def create_abstract_links(abstract: str, name: str) -> str:
    pmid = re.search(PMID_REGEXP, abstract)[1]
    first = rf"\hypertarget{{pmid_{pmid}}}{{{abstract[0]}}}"
    second = re.sub(
        PMID_REGEXP, rf"\\hyperlink{{{name}}}{{PMID: {pmid}}}", abstract[1:]
    )
    return first + second


with open("answers.json", "r") as f:
    answers = json.load(f)

lines = []
for answer in answers:
    # drug name for section
    lines.append(rf"\section*{{{escape_latex_chars(answer['name'])}}}")

    # chatgpt interpretation
    lines.append(r"\subsection*{Result}")
    chatgpt_no_attached_refs = answer["answer"].split("\nReferences:")[0]
    maybe_answer_part = re.search(ANSWER_REGEXP, chatgpt_no_attached_refs)
    maybe_explanation_part = re.search(
        EXPLANATION_REGEXP, chatgpt_no_attached_refs
    )
    lines.append(r"\subsubsection*{Answer}")
    if maybe_answer_part is not None:
        lines.append(escape_latex_chars(maybe_answer_part[1]))
    lines.append(r"\subsubsection*{{Explanation}}")
    if maybe_explanation_part is not None:
        explanation = re.sub(
            PMID_REGEXP,
            r"\\hyperlink{pmid_\1}{PMID: \1}",
            escape_latex_chars(maybe_explanation_part[1]),
        )
        lines.append(
            rf"\hypertarget{{{answer['name']}}}{explanation[0]}"
            + explanation[1:]
        )

    # abstracts
    lines.append(r"\subsection*{Abstracts}")
    split_abstracts = re.split(
        REFERENCE_BOUNDARY_REGEXP, escape_latex_chars(answer["abstracts"])
    )
    abstracts = []
    abstracts.append(
        split_abstracts[0] + split_abstracts[1] + "\n"
    )  # first abstract missing "\. "
    batched_abstracts = list(itertools.batched(split_abstracts[2:-2], 3))
    for abstract in batched_abstracts:
        abstracts.append(abstract[0] + abstract[1] + abstract[2] + "\n")
    abstracts.append(
        split_abstracts[-2] + split_abstracts[-1] + "\n"
    )  # last abstract missing " ([A-Z])"
    abstracts = list(
        map(
            functools.partial(create_abstract_links, name=answer["name"]),
            abstracts,
        )
    )
    lines.extend(abstracts)

whole_document = "\n".join(PREAMBLE + lines + POSTAMBLE)
print(whole_document)
