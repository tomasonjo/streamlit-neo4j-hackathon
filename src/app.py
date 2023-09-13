import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.schema import HumanMessage, AIMessage

st.title("🦜🔗 Quickstart App")

from cypher_chain import CustomCypherChain

url = st.secrets["NEO4J_URL"]
username = st.secrets["NEO4J_USERNAME"]
password = st.secrets["NEO4J_PASSWORD"]
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ['OPENAI_API_KEY'] = openai_api_key

# Langchain x Neo4j connections
graph = Neo4jGraph(username=username, password=password, url=url)

graph_search = CustomCypherChain.from_llm(
    cypher_llm=ChatOpenAI(temperature=0.7, model_name='gpt-4'),
    qa_llm=ChatOpenAI(temperature=0.7),
    graph=graph
)

# Session state
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "user_input" not in st.session_state:
    st.session_state["user_input"] = []

def generate_context(prompt, context_data='generated'):
    context = []
    # If any history exists
    if st.session_state['generated']:
        # Add the last three exchanges
        size = len(st.session_state['generated'])
        for i in range(max(size-3, 0), size):
            context.append(
                HumanMessage(content=st.session_state['user_input'][i]))
            context.append(
                AIMessage(content=st.session_state[context_data][i]))
    # Add the latest user prompt
    context.append(HumanMessage(content=str(prompt)))
    return context


def get_text():
    input_text = st.chat_input("Ask your question")
    return input_text

user_input = get_text()

if user_input:
    context = generate_context(user_input)
    output = graph_search.run(query=user_input, chat_history=context)

    st.session_state.user_input.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    size = len(st.session_state['generated'])
    # Display only the last three exchanges
    for i in range(max(size-3, 0), size):
        with st.chat_message("user"):
            st.write(st.session_state['user_input'][i])
        with st.chat_message("assistant"):
            st.write(st.session_state["generated"][i])