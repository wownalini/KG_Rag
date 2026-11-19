import os
import sys
import certifi
import re
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


if sys.platform == "win32":
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

st.set_page_config(page_title="TN Agri Schemes KG-RAG", page_icon="🌾", layout="wide")

st.title("🌾 Tamil Nadu Agriculture Schemes Assistant")
st.caption("Knowledge Graph RAG powered by Neo4j")

@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def search_graph(driver, query: str):
    keywords = re.findall(r'\w+', query.lower())
    stop_words = {"what", "is", "are", "the", "for", "about", "tell", "me", "schemes", "available"}
    keywords = [k for k in keywords if k not in stop_words and len(k) > 2]
    
    results = []
    with driver.session() as session:
        for kw in keywords:
            cypher = """
                MATCH (s:Scheme)-[:IMPLEMENTED_BY]->(d:Department)
                OPTIONAL MATCH (s)-[:FOR]->(b:Beneficiary)
                OPTIONAL MATCH (s)-[:PROVIDES]->(ben:Benefit)
                WHERE toLower(s.name) CONTAINS toLower($kw)
                   OR toLower(s.description) CONTAINS toLower($kw)
                   OR toLower(d.name) CONTAINS toLower($kw)
                   OR toLower(b.name) CONTAINS toLower($kw)
                RETURN s.name AS name, s.description AS description, 
                       d.name AS department, b.name AS beneficiary, ben.name AS benefit
            """
            records = session.run(cypher, kw=kw)
            for r in records:
                entry = dict(r)
                if entry not in results:
                    results.append(entry)
    return results

def get_answer(question: str):
    driver = get_driver()
    results = search_graph(driver, question)
    
    if not results:
        return "I don't have information about that in the knowledge graph.", []
    
    context = "\n\n".join([
        f"Scheme: {r['name']}\nDepartment: {r['department']}\nDescription: {r['description']}\nBeneficiary: {r['beneficiary']}\nBenefit: {r['benefit']}"
        for r in results
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an assistant for Tamil Nadu agriculture schemes.
Answer the question using ONLY the context below. Be clear and concise.

Context:
{context}

Question: {question}

Answer:"""
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content, results

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about agriculture schemes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge graph..."):
            answer, sources = get_answer(prompt)
            st.write(answer)
            if sources:
                with st.expander("Sources from Knowledge Graph"):
                    for s in sources:
                        st.write(f"**{s['name']}** — {s['department']}")
            st.session_state.messages.append({"role": "assistant", "content": answer})