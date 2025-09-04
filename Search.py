import logging
import math
import os
import re

import numpy as np
import ollama
import tqdm
from numpy.typing import NDArray
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import NamedVector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("embedding_search.log"),
        logging.StreamHandler(),
    ],
)


class PubMedSearcher:
    def __init__(
        self,
        embedding_model: str,
        qdrant_client: QdrantClient,
        qdrant_collection_name: str,
        openai_client: OpenAI,
    ):
        self.embedding_model = embedding_model
        self.qdrant_client = qdrant_client
        self.qdrant_collection_name = qdrant_collection_name
        self.openai_client = openai_client

        self.not_alphanumeric = re.compile(r"[^a-zA-Z0-9]+")

    def search(
        self,
        database_query: str,
        llm_query: str,
        top_n: int = 25,
        llm_model: str = "gpt-4.1",
        *,
        save: bool = True,
        show: bool = True,
    ) -> tuple[str, dict[str, dict], str]:
        embeddings = self.generate_embedding(database_query)
        abstracts, citations = self.search_qdrant(embeddings, top_n, save)
        result = self.generate_answer(
            abstracts, citations, llm_query, llm_model, show
        )
        return abstracts, citations, result

    def generate_embedding(self, query: str) -> NDArray:
        logging.info("Generating embedding for query.")

        response = ollama.embeddings(model=self.embedding_model, prompt=query)
        embedding = np.asarray(response["embedding"])
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def search_qdrant(
        self, embedding: NDArray, top_n: int, save: bool = False
    ) -> tuple[str, dict[str, dict]]:
        logging.info("Searching Qdrant for similar abstracts.")

        search_results = []
        previous_limit = 0
        limit = (
            top_n * 10
        )  # start at rough ratio of sentence embeddings to regular embeddings
        while len(search_results) < top_n:
            all_search_results = self.qdrant_client.search(
                collection_name=self.qdrant_collection_name,
                query_vector=NamedVector(
                    # should end up with the same bgem3_embedding name
                    name=f"{re.sub(self.not_alphanumeric, '', self.embedding_model)}_embedding",
                    vector=embedding.tolist(),
                ),
                limit=limit,
                with_payload=True,
            )

            search_results.extend(
                list(
                    filter(
                        lambda x: x.payload.get("type", None)
                        != "sentence_embedding",
                        all_search_results[previous_limit:],
                    )
                )
            )
            previous_limit = limit
            if len(search_results) > 0:
                limit = math.ceil(limit * top_n / len(search_results))
            else:
                limit = math.ceil(limit * 2)

        # limit search results to the correct number if we overshot
        search_results = search_results[:top_n]

        abstracts_with_citations = []
        citations = {}

        for result in search_results:
            abstract = result.payload.get("abstract", "No abstract available")
            title = result.payload.get("title", "Unknown Title")
            authors = result.payload.get("authors", [])
            pmid = result.payload.get("pmid", "No PMID")

            # Extract year safely from journal
            journal_info = result.payload.get("journal", {})
            pub_date = journal_info.get("PubDate", {})
            year = pub_date.get("Year", "Unknown Year")

            # Extract author names 
            if isinstance(authors, list):
                author_names = [
                    f"{author.get('ForeName', '')} {author.get('LastName', '')}".strip()
                    for author in authors
                    if isinstance(author, dict)
                ]
            else:
                author_names = []

            # Format citation key
            if author_names:
                first_author = author_names[0]
                citation_key = f"[PMID: {pmid}, {first_author} et al., {year}]"
            else:
                citation_key = f"[PMID: {pmid}, {title}, {year}]"

            # Store citation details
            citations[citation_key] = {
                "title": title,
                "authors": ", ".join(author_names)
                if author_names
                else "Unknown Authors",
                "year": year,
                "pmid": pmid,  
            }

            # Append abstract with citation marker
            abstracts_with_citations.append(f"{abstract} {citation_key}")

        combined_abstracts = " ".join(abstracts_with_citations)

        if save:
            with open("abstracts_output.txt", "w", encoding="utf-8") as file:
                file.write(combined_abstracts)

        return combined_abstracts, citations

    def generate_answer(
        self,
        abstracts: str,
        citations: dict[str, dict],
        query: str,
        model: str,
        show: bool = False,
    ) -> str:
        if not abstracts:
            return "I'm sorry, but I couldn't find a relevant abstract for your question."

        logging.info("Generating answer with citations.")

        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable assistant who provides information based on scientific abstracts. When referencing sources, include in-text citations using the format [PMID: PMID, Author et al., Year] or [PMID: PMID, Title, Year] if the author is unknown.",
            },
            {"role": "user", "content": query},
            {"role": "assistant", "content": abstracts},
        ]

        completion = self.openai_client.chat.completions.create(
            model=model, messages=messages, temperature = 0
        )
        answer = completion.choices[0].message.content

        # Replace DOI with PMID in the reference section
        reference_section = "\n\nReferences:\n" + "\n".join(
            [
                f"{key}: {value['title']} ({value['year']}), PMID: {value['pmid']}"
                for key, value in citations.items()
            ]
        )

        final_answer = answer + reference_section
        if show:
            print(final_answer)
        return final_answer


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Qdrant client setup
    qdrant_client = QdrantClient(host="localhost", port=6333, timeout = 300.0)
    collection_name = "PubMed"

    # OpenAI client setup
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    openAI_client = OpenAI(api_key=api_key)

    searcher = PubMedSearcher(
        "bge-m3", qdrant_client, collection_name, openAI_client
    )

    # Process inputs from file
    import json

    with open("drug_names.txt", "r") as f:
        drugs = map(lambda s: s.split("\n")[0], f.readlines())

    json_output = []
    for drug in tqdm.tqdm(drugs):
        output = {"name": drug}
        database_query = f"safety of {drug} used in children"
        llm_query = f"""Using the abstracts you have for {drug}, determine if {drug} is safe ('no toxicity', 'no adverse events') for use in children (17 years and younger). 
        Your explanation should be evidence based and only represent what you can find in the abstracts. 
        Safe for use in children means that a targeted study has been done about safety in children and that the study affirms its safety.
        Do not extrapolate safety in the general population or adults to mean safe in children. 
        Not safe for use in children means the opposite: a targeted study has been done about safety in children, and that study shows {drug} is unsafe (results in 'adverse events' or 'toxicity'). 
        If there is no data to definitively prove safe or unsafe, then that means the safety is unknown.
        If the abstracts only mention "children" but do not specify ages, then that means safety is unknown.
        You should summarize the relevant abstracts into an answer using specific age ranges. 
        Do not use your own knowledge to make educated guesses.
        When citing, include in-text citations using the format [PMID: PMID, Author et al., Year] or [PMID: PMID, Title, Year] if the author is unknown.
        Use '<explain>' and '</explain>' tags around your explanation first, then use '<answer>' and '</answer>' tags to contain either 'Yes', 'No', or 'Unknown' for each range identified. 
        If none of the abstracts you have are about {drug}, then your explanation should reflect that not enough data was available to you."""
        output["abstracts"], output["citations"], output["answer"] = (
            searcher.search(
                database_query,
                llm_query,
                llm_model="gpt-4.1",
                save=True,
                show=False,
            )
        )
        json_output.append(output)

    with open("answers.json", "w") as f:
        json.dump(json_output, f)
