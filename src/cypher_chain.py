import os

from typing import Any, Dict, List, Optional, Tuple
from langchain.chains.graph_qa.cypher import extract_cypher
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate

from langchain.chains import GraphCypherQAChain
from langchain.callbacks.manager import CallbackManagerForChainRun

try:
    from pydantic.v1.main import BaseModel, Field
except ImportError:
    from pydantic.main import BaseModel, Field

from cypher_validator import CypherValidator


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
        new_tokens.extend([token.text for token in doc[last_end : ent.start]])
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

CYPHER_SYSTEM_TEMPLATE = """
Your task is to convert questions about contents in a Neo4j database to Cypher queries to query the Neo4j database.
Use only the provided relationship types and properties.
Do not use any other relationship types or properties that are not provided.
"""

validator = CypherValidator()

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Even if the question doesn't provide full person or organization names, you should use the full names from the provided
information to construct an answer.
Information:
{context}

Question: {question}
Helpful Answer:"""
CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)


class Entities(BaseModel):
    """Identifying information about entities."""

    name: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )


class CustomCypherChain(GraphCypherQAChain):
    def process_entities(self, text: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organization and person entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following input: {input}",
                ),
            ]
        )

        entity_chain = create_structured_output_chain(
            Entities, self.qa_chain.llm, prompt
        )
        entities = entity_chain.run(text)
        print(entities)
        return entities.name

    def get_viz_data(self, entities: List[str]) -> List[Tuple[str, str]]:
        viz_query = """
        MATCH (n:Person|Organization) WHERE n.name IN $entities
        CALL {
            WITH n
            MATCH (n)-[r:!MENTIONS]->(m)
            WHERE m.name IS NOT NULL
            RETURN n.name AS source, type(r) AS type, m.name AS target
            LIMIT 5
            UNION
            WITH n
            MATCH (n)<-[r:!MENTIONS]-(m)
            WHERE m.name IS NOT NULL
            RETURN n.name AS target, type(r) AS type, m.name AS source
            LIMIT 5
        }
        RETURN source, type, target LIMIT 20
        """
        results = self.graph.query(viz_query, {"entities": entities})
        return results

    def find_entity_match(self, entity: str, k: int = 3) -> List[str]:
        fts_query = """
        CALL db.index.fulltext.queryNodes('entity', $entity + "*", {limit:$k})
        YIELD node,score
        RETURN node.name AS result
        """

        return [
            el["result"]
            for el in self.graph.query(
                fts_query, {"entity": "AND ".join(entity.split()), "k": k}
            )
        ]

    def generate_system_message(
        self, relevant_entities: str = "", fewshot_examples: str = ""
    ) -> SystemMessage:
        system_message = CYPHER_SYSTEM_TEMPLATE
        system_message += (
            f"The database has the following schema: {self.graph.get_schema} "
        )
        if relevant_entities:
            system_message += (
                f"Relevant entities for the question are: {relevant_entities} "
                "Always replace the entity in the input question with relevant entites from the list\n"
                "For example, if the relevant entities mention a person: John : ['John Goodman', 'John Stockton'] "
                "You should always use a query that catch all the available options. If you want to have a query like: "
                "Template: 'MATCH (p:Person {name:'John'})<-[:BOARD_MEMBER]-(o:Organization)' You need to split it into two MATCHES"
                "Corrected query: 'MATCH (p:Person) WHERE p.name IN ['John Goodman', 'John Stockton'] "
                "MATCH (p)<-[:BOARD_MEMBER]-(o:Organization)' "

            )
        if fewshot_examples:
            system_message += f"Follow these Cypher examples when you are constructing a Cypher statement: {fewshot_examples} "

        system_message += (
            "Always provide enough information in the response so that an outsider"
            "without any additional context can answer the question. For example, "
            "if the question mentions a person and an organization, you should try to "
            "return both information about the person as well as the organization!"
            "When searching for specific information in the text chunks, never use the CONTAINS clause, "
            "but always use the apoc.ml.openai.embedding and gds.similarity.cosine functions "
            "or db.index.vector.queryNodes as shown in the examples. "
            "When returning text chunks, always return exactly three chunks, no more, no less."
        )
        return SystemMessage(content=system_message)

    def get_fewshot_examples(self, question):
        results = self.graph.query(
            """
        CALL apoc.ml.openai.embedding([$question], $openai_api_key)
                                    YIELD embedding                             
        CALL db.index.vector.queryNodes('fewshot', 3, embedding)
                                    YIELD node, score
        RETURN node.Question AS question, node.Cypher as cypher
                                    """,
            {"question": question, "openai_api_key": os.environ["OPENAI_API_KEY"]},
        )

        return "\n".join([f"#{el['question']}\n{el['cypher']}" for el in results])

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]
        chat_history = inputs["chat_history"]
        intermediate_steps: List = []
        # Extract mentioned people and organizations and match them to database values
        entities = self.process_entities(question)
        print(f"NER found: {entities}")
        relevant_entities = dict()
        for entity in entities:
            relevant_entities[entity] = self.find_entity_match(entity)
        print(f"Relevant entities are: {relevant_entities}")

        # Get few-shot examples using vector search
        fewshots = self.get_fewshot_examples(question)

        system = self.generate_system_message(str(relevant_entities), fewshots)
        generated_cypher = self.cypher_generation_chain.llm.predict_messages(
            [system] + chat_history
        )
        print(generated_cypher.content)
        generated_cypher = extract_cypher(generated_cypher.content)
        validated_cypher = validator.validate_query(
            AVAILABLE_RELATIONSHIPS, generated_cypher
        )
        print(validated_cypher)
        # If Cypher statement wasn't generated
        # Usually happens when LLM decides it can't answer
        if not "RETURN" in validated_cypher[0]:
            chain_result: Dict[str, Any] = {
                self.output_key: validated_cypher[0],
                "viz_data": (None, None),
                "database": None,
                "cypher": None,
            }
            return chain_result

        # Retrieve and limit the number of results
        context = self.graph.query(
            validated_cypher[0], {"openai_api_key": os.environ["OPENAI_API_KEY"]}
        )[: self.top_k]

        result = self.qa_chain(
            {"question": question, "context": context}, callbacks=callbacks
        )
        final_result = result[self.qa_chain.output_key]

        final_entities = self.process_entities(final_result)
        if final_entities:
            viz_data = self.get_viz_data(final_entities)
        else:
            viz_data = None

        chain_result: Dict[str, Any] = {
            self.output_key: final_result,
            "viz_data": (viz_data, final_entities),
            "database": context,
            "cypher": validated_cypher[0],
        }
        return chain_result
