import os
import re
import sys
import certifi
from dotenv import load_dotenv
from neo4j import GraphDatabase
import json

if sys.platform == "win32":
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Sample schemes data
# Load schemes from JSON file
with open("schemes.json", "r", encoding="utf-8") as f:
    schemes = json.load(f)

print(f"Loaded {len(schemes)} schemes")

def clear_graph(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def push_schemes_to_graph(driver):
    with driver.session() as session:
        for s in schemes:
            cypher = """
                MERGE (scheme:Scheme {name: $name})
                SET scheme.description = $description
                MERGE (dept:Department {name: $department})
                MERGE (ben:Beneficiary {name: $beneficiary})
                MERGE (benefit:Benefit {name: $benefit})
                MERGE (scheme)-[:IMPLEMENTED_BY]->(dept)
                MERGE (scheme)-[:FOR]->(ben)
                MERGE (scheme)-[:PROVIDES]->(benefit)
            """
            session.run(cypher, **s)
    print(f"Pushed {len(schemes)} schemes to Neo4j")

def search_graph(driver, query: str):
    keywords = re.findall(r'\w+', query.lower())
    stop_words = {"what", "is", "are", "the", "for", "about", "tell", "me", "schemes"}
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
                RETURN s.name AS name, s.description AS description, 
                       d.name AS department, b.name AS beneficiary, ben.name AS benefit
            """
            records = session.run(cypher, kw=kw)
            for r in records:
                entry = dict(r)
                if entry not in results:
                    results.append(entry)
    return results

def create_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an assistant for Tamil Nadu agriculture schemes.
Answer using ONLY the context below.

Context:
{context}

Question: {question}

Answer:"""
    )
    return prompt | llm

def main():
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        driver.verify_connectivity()
        print("Connected!")
        
        clear_graph(driver)
        push_schemes_to_graph(driver)
        
        chain = create_chain()
        
        print("\nAsk about agriculture schemes (type 'quit' to exit)")
        while True:
            q = input("\nYou: ").strip()
            if q.lower() in ("quit", "exit"):
                break
            if not q:
                continue
                
            results = search_graph(driver, q)
            context = "\n".join([str(r) for r in results]) if results else "No relevant schemes found."
            
            response = chain.invoke({"context": context, "question": q})
            print(f"\nAssistant: {response.content}")
            
    except Exception as e:
        print("Error:", e)
    finally:
        driver.close()

if __name__ == "__main__":
    main()