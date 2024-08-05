import streamlit as st
from llm import llm
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Create the Neo4jVector index
existing_index = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=st.secrets["NEO4J_URI"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
    index_name="wikipedia",
    text_node_property="info",
    retrieval_query="""
    OPTIONAL MATCH (node)<-[:INCLUDES]-(p)
    WITH node, score, collect(p) AS editors
    RETURN node.info AS text,
        score,
        node {.*, vector: Null, info: Null, editors: editors} AS metadata

    """
)


# Create the retriever
retriever = existing_index.as_retriever()

# Create the prompt
instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

# Create the chain 
question_answer_chain = create_stuff_documents_chain(llm, prompt)
plot_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

# Create a function to call the chain
def get_vector_info(input):
    return plot_retriever.invoke({"input": input})