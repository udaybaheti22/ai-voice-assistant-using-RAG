from rag.build_package import load_package
from rag.retrieve import retrieve
from rag.llm import generate_response

def build_prompt(query, context_chunks):
    context = "\n\n".join([c["chunk"] for c in context_chunks])

    prompt = f"""
You are an intelligent assistant.

Answer the question ONLY using the context below.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt


def run_rag(query, package="finance"):
    index, chunks = load_package(package)
    
    results = retrieve(query, index, chunks, k=3)
    
    prompt = build_prompt(query, results)
    
    answer = generate_response(prompt)
    
    return answer