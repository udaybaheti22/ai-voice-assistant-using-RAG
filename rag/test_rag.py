from rag.build_package import load_package
from rag.retrieve import retrieve

index, chunks = load_package("finance")

print("Total chunks:", len(chunks))
print("Index size:", index.ntotal)

query = "What is finance?"

results = retrieve(query, index, chunks, k=3)

print("Results:", results)

for r in results:
    print("\n---")
    print(r["chunk"])