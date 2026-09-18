import chromadb
from sentence_transformers import SentenceTransformer

# Load the same embedding model used to build the index
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to the same ChromaDB we already built
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="policies")

# The question we want to ask
question = "How many vacation days do I get?"

# Embed the question the same way we embedded the documents
question_embedding = embedder.encode([question]).tolist()

# Search ChromaDB for the most similar document(s)
results = collection.query(
    query_embeddings=question_embedding,
    n_results=2  # return the top 2 closest matches
)

print(f"Question: {question}\n")
print("Top matching documents:")
for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
    print(f"- {doc_id} (distance: {distance:.4f})")

print("\nBest match content:")
print(results["documents"][0][0])
