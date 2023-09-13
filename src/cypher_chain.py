from typing import Any, Dict, List, Optional
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT
from langchain.chains.graph_qa.cypher import extract_cypher



import streamlit as st
import spacy
from langchain.chains import GraphCypherQAChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel

from cypher_validator import CypherValidator


nlp = spacy.load("en_core_web_md")
validator = CypherValidator()

def remove_entities(doc):
    """
    Replace named entities in the given text with their corresponding entity labels.

    Parameters:
    - doc (Spacy Document): processed SpaCy document of the input text.

    Returns:
    - str: The modified text with named entities replaced by their entity labels.

    Example:
    >>> replace_entities_with_labels("Apple is looking at buying U.K. startup for $1 billion.")
    'ORG is looking at buying GPE startup for MONEY .'
    """
    # Initialize an empty list to store the new tokens
    new_tokens = []
    # Keep track of the end index of the last entity
    last_end = 0

    # Iterate through entities, replacing them with their entity label
    for ent in doc.ents:
        # Add the tokens that come before this entity
        new_tokens.extend([token.text for token in doc[last_end:ent.start]])
        # Replace the entity with its label
        new_tokens.append(f"{ent.label_}")
        # Update the last entity end index
        last_end = ent.end

    # Add any remaining tokens after the last entity
    new_tokens.extend([token.text for token in doc[last_end:]])
    # Join the new tokens into a single string
    new_text = " ".join(new_tokens)
    return new_text


AVAILABLE_RELATIONSHIPS = """
    (Person, HAS_PARENT, Person),
    (Person, HAS_CHILD, Person),
    (Organization, HAS_SUPPLIER, Organization),
    (Organization, IN_CITY, City),
    (Organization, HAS_CATEGORY, IndustryCategory),
    (Organization, HAS_CEO, Person),
    (Organization, HAS_SUBSIDIARY, Organization),
    (Organization, HAS_COMPETITOR, Organization),
    (Organization, HAS_BOARD_MEMBER, Person),
    (Organization, HAS_INVESTOR, Organization),
    (Organization, HAS_INVESTOR, Person),
    (City, IN_COUNTRY, Country),
    (Article, HAS_CHUNK, Chunk),
    (Article, MENTIONS, Organization)
"""

class CustomCypherChain(GraphCypherQAChain):
    def find_entity_match(self, entity:str, k: int = 3):
        fts_query = """
        CALL db.index.fulltext.queryNodes('entity', $entity + "*", {limit:$k})
        YIELD node,score
        RETURN node.name AS result
        """

        return [el['result'] for el in self.graph.query(fts_query, {'entity': "AND ".join(entity.split()), 'k':k})]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        print(inputs)
        question = inputs[self.input_key]
        chat_history = inputs['chat_history']
        intermediate_steps: List = []


        spacy_doc = nlp(question)

        # Extract mentioned people and organizations and match them to database values
        entities = [{'text':ent.text, 'label': ent.label_} for ent in spacy_doc.ents if ent.label_ in ["PERSON", "ORG"]]
        print(f"SpaCy found: {entities}")
        relevant_entities = dict()
        for entity in entities:
            relevant_entities[entity['text']] = self.find_entity_match(entity['text'])
        print(f"Relevant etities are: {relevant_entities}")

        # Get few-shot examples using vector search
        cleaned_question = remove_entities(spacy_doc)
        fewshots = "vectorsearch"

        generated_cypher = self.cypher_generation_chain.run(
            {
                "question": question,
                "schema": self.graph.get_schema,
                "few-shot": fewshots,
                "relevant_entities": relevant_entities,
            },
            callbacks=callbacks,
        )
        generated_cypher = extract_cypher(generated_cypher)
        validated_cypher = validator.validate_query(AVAILABLE_RELATIONSHIPS, generated_cypher)
        print(validated_cypher)
        # Retrieve and limit the number of results
        context = self.graph.query(validated_cypher[0])[: self.top_k]

        result = self.qa_chain(
                {"question": question, "context": context},
                callbacks=callbacks,
            )
        final_result = result[self.qa_chain.output_key]
        chain_result: Dict[str, Any] = {self.output_key: final_result}
        return chain_result