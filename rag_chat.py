import requests
import chromadb
from sentence_transformers import SentenceTransformer

# Load the embedding model (same one used to build the index)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to the existing ChromaDB collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="policies")

def retrieve_context(question, n_results=8, distance_threshold=1.5):
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=n_results)

    chunks = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]

    print(f"DEBUG — question: '{question}'")
    for doc_id, distance in zip(ids, distances):
        print(f"   {doc_id}: distance={distance:.4f}")

    filtered_chunks, filtered_ids = [], []
    for chunk, doc_id, distance in zip(chunks, ids, distances):
        if distance <= distance_threshold:
            filtered_chunks.append(chunk)
            filtered_ids.append(doc_id)

    return filtered_chunks, filtered_ids

def ask_llm(question, context_chunks):
    """Build a grounded prompt and send it to the local LLM."""
    context_text = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant answering questions based only on the provided company policy documents.
If the answer isn't in the provided context, say you don't have that information — do not guess.

Context:
{context_text}

Question: {question}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    return response.json()["message"]["content"]

# Interactive loop
print("RAG Chatbot (type 'quit' to exit)")
print("-" * 40)

while True:
    question = input("\nYou: ")
    if question.lower() == "quit":
        break

    chunks, ids, distances = retrieve_context(question)

    print(f"\n[Retrieved from: {ids}]")
    answer = ask_llm(question, chunks)
    print(f"\nAssistant: {answer}")