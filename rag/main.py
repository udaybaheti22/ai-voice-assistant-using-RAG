from rag.rag_pipeline import run_rag

while True:
    query = input("You: ")
    
    if query.lower() in ["exit", "quit"]:
        break
    
    answer = run_rag(query)
    
    print("\nAI:", answer)