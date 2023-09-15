import os
from typing import List, Union

import streamlit as st
import graphviz
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.schema import HumanMessage, AIMessage

st.title("VC Chatbot")

from cypher_chain import CustomCypherChain

url = st.secrets["NEO4J_URL"]
username = st.secrets["NEO4J_USERNAME"]
password = st.secrets["NEO4J_PASSWORD"]

# Langchain x Neo4j connections
graph = Neo4jGraph(username=username, password=password, url=url)

graph_search = None

# Session state
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "user_input" not in st.session_state:
    st.session_state["user_input"] = []

if "viz_data" not in st.session_state:
    st.session_state["viz_data"] = []

if "database" not in st.session_state:
    st.session_state["database"] = []

if "cypher" not in st.session_state:
    st.session_state["cypher"] = []


def generate_context(
    prompt: str, context_data: str = "generated"
) -> List[Union[AIMessage, HumanMessage]]:
    context = []
    # If any history exists
    if st.session_state["generated"]:
        # Add the last three exchanges
        size = len(st.session_state["generated"])
        for i in range(max(size - 3, 0), size):
            context.append(HumanMessage(content=st.session_state["user_input"][i]))
            context.append(AIMessage(content=st.session_state[context_data][i]))
    # Add the latest user prompt
    context.append(HumanMessage(content=str(prompt)))
    return context


def dynamic_response_tabs(i):
    tabs_to_add = ["💬Chat"]
    data_check = {
        "🔍Cypher": st.session_state["cypher"][i],
        "🗃️Database results": st.session_state["database"][i],
        "🕸️Visualization": st.session_state["viz_data"][i]
        and st.session_state["viz_data"][i][0],
    }

    for tab_name, has_data in data_check.items():
        if has_data:
            tabs_to_add.append(tab_name)

    with st.chat_message("user"):
        st.write(st.session_state["user_input"][i])

    with st.chat_message("assistant"):
        selected_tabs = st.tabs(tabs_to_add)

        with selected_tabs[0]:
            st.write(st.session_state["generated"][i])
        if len(selected_tabs) > 1:
            with selected_tabs[1]:
                st.code(st.session_state["cypher"][i], language="cypher")
        if len(selected_tabs) > 2:
            with selected_tabs[2]:
                st.write(st.session_state["database"][i])
        if len(selected_tabs) > 3:
            with selected_tabs[3]:
                graph_object = graphviz.Digraph()
                for final_entity in st.session_state["viz_data"][i][1]:
                    graph_object.node(
                        final_entity, fillcolor="lightblue", style="filled"
                    )
                for record in st.session_state["viz_data"][i][0]:
                    graph_object.edge(
                        record["source"], record["target"], label=record["type"]
                    )
                st.graphviz_chart(graph_object)


def get_text() -> str:
    input_text = st.chat_input("Who is the CEO of Neo4j?")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    else:
        return input_text


openai_api_key = st.sidebar.text_input("OpenAI API Key")
os.environ["OPENAI_API_KEY"] = openai_api_key
if openai_api_key:
    graph_search = CustomCypherChain.from_llm(
        cypher_llm=ChatOpenAI(temperature=0.0, model_name="gpt-4"),
        qa_llm=ChatOpenAI(temperature=0.0),
        graph=graph,
    )
st.sidebar.markdown(
    """
## Example questions

* What do you know about Neo4j organization?
* Who is Neo4j CEO?
* How is Emil Eifrem connected to Magnus Christerson?
* Which company has the most subsidiaries?
* What are the latest news around companies where Emil Eifrem is CEO?
* Are there any news about new partnerships mentioning Neo4j?
* Are there any partnerships mentioned in news for companies where Daniel Rumennik is an investor? 

You can also ask follow up questions as we use a conversational LLM under the hood.

Code is available on [GitHub](https://github.com/tomasonjo/streamlit-neo4j-hackathon)
"""
)

user_input = get_text()

if user_input:
    context = generate_context(user_input)
    output = graph_search({"query": user_input, "chat_history": context})

    st.session_state.user_input.append(user_input)
    st.session_state.generated.append(output["result"])
    st.session_state.viz_data.append(output["viz_data"])
    st.session_state.database.append(output["database"])
    st.session_state.cypher.append(output["cypher"])

if st.session_state["generated"]:
    size = len(st.session_state["generated"])
    # Display only the last three exchanges
    for i in range(max(size - 3, 0), size):
        dynamic_response_tabs(i)
