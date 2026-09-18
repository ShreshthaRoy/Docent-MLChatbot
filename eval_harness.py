import csv
import time as timer
import chromadb
from sentence_transformers import SentenceTransformer
import requests

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="policies")


def retrieve_context(question, n_results=2, distance_threshold=1.5):
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=n_results)

    chunks, ids, distances = results["documents"][0], results["ids"][0], results["distances"][0]
    filtered_chunks, filtered_ids = [], []
    for chunk, doc_id, distance in zip(chunks, ids, distances):
        if distance <= distance_threshold:
            filtered_chunks.append(chunk)
            filtered_ids.append(doc_id)
    return filtered_chunks, filtered_ids


def ask_llm(question, context_chunks, model="llama3.2"):
    if not context_chunks:
        return "I don't have any information relevant to that question.", 0.0

    context_text = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant answering questions based only on the provided company policy documents.
If the answer isn't in the provided context, say you don't have that information — do not guess.

Context:
{context_text}

Question: {question}

Answer:"""

    start = timer.time()
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
        timeout=180
    )
    elapsed = timer.time() - start

    return response.json()["message"]["content"], elapsed


def run_evaluation(csv_path="eval_questions.csv", model="llama3.2"):
    results = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        test_cases = list(reader)

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        expected_source = case["expected_source"].strip()
        expected_keyword = case["expected_keyword"].strip()

        print(f"[{i}/{len(test_cases)}] Testing: {question}")

        chunks, ids = retrieve_context(question)
        retrieved_sources = [doc_id.split("::")[0] for doc_id in ids]

        answer, elapsed = ask_llm(question, chunks, model=model)

        if expected_source == "NONE":
            retrieval_correct = len(retrieved_sources) == 0
        else:
            retrieval_correct = expected_source in retrieved_sources

        if expected_source == "NONE":
            answer_correct = "don't have" in answer.lower() or "not specified" in answer.lower() or "no information" in answer.lower()
        elif expected_keyword:
            answer_correct = expected_keyword.lower() in answer.lower()
        else:
            answer_correct = True

        results.append({
            "question": question,
            "retrieval_correct": retrieval_correct,
            "answer_correct": answer_correct,
            "elapsed": elapsed,
            "retrieved_sources": retrieved_sources,
            "answer": answer
        })

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT — model: {model}")
    print("=" * 60)

    retrieval_passed = sum(1 for r in results if r["retrieval_correct"])
    answer_passed = sum(1 for r in results if r["answer_correct"])
    avg_time = sum(r["elapsed"] for r in results) / len(results)
    total_time = sum(r["elapsed"] for r in results)

    print(f"Retrieval correct: {retrieval_passed}/{len(results)}")
    print(f"Answer correct:    {answer_passed}/{len(results)}")
    print(f"Average response time: {avg_time:.1f}s")
    print(f"Total time: {total_time:.1f}s\n")

    for r in results:
        retrieval_icon = "✅" if r["retrieval_correct"] else "⚠️ "
        answer_icon = "✅" if r["answer_correct"] else "❌"
        print(f"{answer_icon} Answer | {retrieval_icon} Retrieval | {r['elapsed']:.1f}s — {r['question']}")

    return results


if __name__ == "__main__":
    print("\n########## TESTING llama3.2 ##########")
    run_evaluation(model="llama3.2")

    

    print("\n\n########## TESTING phi4-mini ##########")
    run_evaluation(model="phi4-mini")

    print("\n\n########## TESTING qwen2.5:0.5b ##########")
    run_evaluation(model="qwen2.5:0.5b")