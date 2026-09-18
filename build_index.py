import os
import chromadb
from sentence_transformers import SentenceTransformer

# Step 1: Load the embedding model (pretrained, converts text -> vectors)
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Set up ChromaDB (our vector database)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="policies")

# Step 3: Read all documents from dummy_docs folder
docs_folder = "dummy_docs"
documents = []
doc_ids = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(docs_folder, filename)
        with open(filepath, "r") as f:
            content = f.read()
            documents.append(content)
            doc_ids.append(filename)

print(f"Loaded {len(documents)} documents: {doc_ids}")

# Step 4: Embed each document and store in ChromaDB
embeddings = embedder.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=doc_ids
)

print("Documents embedded and stored successfully!")
print(f"Collection now has {collection.count()} items.")
