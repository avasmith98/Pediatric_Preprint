import re


def check_citations(llm_output: str, citations: dict[str, dict]) -> set | None:
    pmid_regex = re.compile(r"PMID: (\d+)")
    output_no_refs = llm_output.split("\nReferences:")[0]

    referenced_pmids = set(re.findall(pmid_regex, output_no_refs))
    citation_pmids = set(
        map(lambda citation: citation["pmid"], citations.values())
    )

    difference = referenced_pmids - citation_pmids
    return difference if difference else None


if __name__ == "__main__":
    import json

    with open("answers.json", "r") as f:
        output = json.load(f)

    for drug in output:
        diff = check_citations(drug["answer"], drug["citations"])
        if diff is not None:
            print(f"{drug} mismatched PMIDs: {diff}")
