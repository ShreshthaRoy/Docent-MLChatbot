import streamlit as st
import requests
import time
import re
import csv
import json
import uuid
import numpy as np
from chat_store import load_all_chats, save_chat, delete_chat
from datetime import datetime
import chromadb
from logger import log_interaction
from sentence_transformers import SentenceTransformer
import os
from build_index_multiformat import READERS, chunk_text

st.set_page_config(page_title="Company Policy Assistant", page_icon="💬", layout="wide")

st.markdown("""
<style>
    .stChatMessage {
        border-radius: 12px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #E0E0E0;
    }
    h1 {
        font-size: 1.8rem !important;
    }
    .block-container {
        padding-bottom: 6rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Identify the user before showing anything else ---
# Lightweight identity check — NOT real authentication. This keeps each
# person's chat history separated in normal use; a real access-control
# layer (login, permissions) is a separate, later piece of work.

if "user_id" not in st.session_state:
    st.title("💬 Company Policy Assistant")
    st.markdown("Please enter your name or email to continue.")
    entered_id = st.text_input("Your name or email", key="login_input")
    if st.button("Continue"):
        if entered_id.strip():
            st.session_state.user_id = entered_id.strip()
            st.rerun()
        else:
            st.warning("Please enter your name or email.")
    st.stop()

user_id = st.session_state.user_id


# --- Load heavy resources ONCE, not on every message ---

@st.cache_resource
def load_embedder():
    import torch
    print("Loading embedder — should only print ONCE per session")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer('all-MiniLM-L6-v2', device=device)

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="policies")

embedder = load_embedder()
collection = load_collection()


# --- Core functions ---

def add_manual_knowledge(text, submitted_by="unknown"):
    chunk_id = f"user_added::{int(time.time())}"
    tagged_text = f"[UNVERIFIED USER-SUBMITTED NOTE — submitted by {submitted_by}]: {text}"
    embedding = embedder.encode([tagged_text]).tolist()
    collection.add(documents=[tagged_text], embeddings=embedding, ids=[chunk_id])
    return chunk_id


def retrieve_context(question, n_results=8, distance_threshold=1.5):
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=n_results)

    chunks = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]

    filtered_chunks, filtered_ids = [], []
    for chunk, doc_id, distance in zip(chunks, ids, distances):
        if distance <= distance_threshold:
            filtered_chunks.append(chunk)
            filtered_ids.append(doc_id)

    return filtered_chunks, filtered_ids

@st.cache_data(ttl=3600)
def retrieve_context_cached(question, n_results=8, distance_threshold=1.5):
    return retrieve_context(question, n_results, distance_threshold)


def log_feedback(question, answer, rating):
    file_exists = os.path.isfile("feedback_log.csv")
    with open("feedback_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "rating"])
        writer.writerow([datetime.now().isoformat(), question, answer, rating])


def find_supporting_sentence(answer, source_chunk):
    sentences = re.split(r'(?<=[.!?])\s+', source_chunk.strip())
    sentences = [s for s in sentences if len(s) > 10]
    if not sentences:
        return None

    sentence_embeddings = embedder.encode(sentences).tolist()
    answer_embedding = embedder.encode([answer]).tolist()[0]

    sentence_vecs = np.array(sentence_embeddings)
    answer_vec = np.array(answer_embedding)
    similarities = sentence_vecs @ answer_vec / (
        np.linalg.norm(sentence_vecs, axis=1) * np.linalg.norm(answer_vec)
    )
    best_idx = int(np.argmax(similarities))
    return sentences[best_idx]


def ask_llm_stream(question, context_chunks):
    context_text = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant answering questions based only on the provided company policy documents.
If the answer isn't in the provided context, say you don't have that information — do not guess.
Answer directly and concisely. Do not discuss the source or verification status of the information in your answer — that is shown separately to the user.
If multiple context entries give conflicting values for the same thing, do not refuse to answer — instead, list each value along with who submitted it, exactly as attributed in the context (e.g. "email@x.com submitted: 1234").

Context:
{context_text}

Question: {question}

Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "keep_alive": "30m",
                "options": {
                    "num_predict": 250
                }
            },
            stream=True,
            timeout=180
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = line.decode("utf-8")
                data = json.loads(chunk)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]

    except requests.exceptions.ConnectionError:
        yield "⚠️ I couldn't reach the AI model. Please make sure Ollama is running and try again."
    except requests.exceptions.Timeout:
        yield "⚠️ The request took too long and timed out. Please try again."
    except requests.exceptions.RequestException as e:
        yield f"⚠️ Something went wrong while generating a response: {e}"
    except Exception as e:
        yield f"⚠️ An unexpected error occurred: {e}"


def render_sources(container, chunks, ids, answer, key_prefix):
    with container:
        with st.expander(f"{len(ids)} source(s) used", expanded=True):
            for idx, (chunk, doc_id) in enumerate(zip(chunks, ids)):
                label = "⚠️ Unverified note" if doc_id.startswith("user_added") else doc_id
                st.markdown(f"**{label}**")

                button_key = f"show_source_{key_prefix}_{doc_id}_{idx}"
                if st.button("🔍 Show supporting sentence", key=button_key):
                    best_sentence = find_supporting_sentence(answer, chunk)
                    if best_sentence:
                        st.markdown(f"> {best_sentence}")
                    else:
                        st.text(chunk[:300] + ("..." if len(chunk) > 300 else ""))
                st.divider()


# --- Multi-chat state, private to this user, loaded from disk once per session ---

if "chats" not in st.session_state:
    st.session_state.chats = load_all_chats(user_id)

if "current_chat_id" not in st.session_state:
    if st.session_state.chats:
        most_recent_id = max(
            st.session_state.chats,
            key=lambda cid: st.session_state.chats[cid].get("updated_at", "")
        )
        st.session_state.current_chat_id = most_recent_id
    else:
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {
            "title": "New chat",
            "messages": [],
            "updated_at": datetime.now().isoformat()
        }
        st.session_state.current_chat_id = new_id
        save_chat(user_id, new_id, st.session_state.chats[new_id])


def current_chat():
    return st.session_state.chats[st.session_state.current_chat_id]


def persist_current_chat():
    current_chat()["updated_at"] = datetime.now().isoformat()
    save_chat(user_id, st.session_state.current_chat_id, current_chat())


# --- Sidebar ---

with st.sidebar:
    st.caption(f"Signed in as **{user_id}**")
    if st.button("Switch user", use_container_width=True):
        for key in ["user_id", "chats", "current_chat_id"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.divider()

    if st.button("➕ New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {
            "title": "New chat",
            "messages": [],
            "updated_at": datetime.now().isoformat()
        }
        st.session_state.current_chat_id = new_id
        save_chat(user_id, new_id, st.session_state.chats[new_id])
        st.rerun()

    st.markdown("### 🕘 Chats")
    sorted_chat_ids = sorted(
        st.session_state.chats.keys(),
        key=lambda cid: st.session_state.chats[cid].get("updated_at", ""),
        reverse=True
    )
    for chat_id in sorted_chat_ids:
        chat_entry = st.session_state.chats[chat_id]
        is_active = chat_id == st.session_state.current_chat_id
        label = ("🔵 " if is_active else "") + chat_entry["title"]

        col_select, col_delete = st.columns([4, 1])
        with col_select:
            if st.button(label, key=f"chat_select_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with col_delete:
            if st.button("🗑️", key=f"chat_delete_{chat_id}"):
                delete_chat(user_id, chat_id)
                del st.session_state.chats[chat_id]

                if chat_id == st.session_state.current_chat_id:
                    if st.session_state.chats:
                        st.session_state.current_chat_id = max(
                            st.session_state.chats,
                            key=lambda cid: st.session_state.chats[cid].get("updated_at", "")
                        )
                    else:
                        new_id = str(uuid.uuid4())
                        st.session_state.chats[new_id] = {
                            "title": "New chat",
                            "messages": [],
                            "updated_at": datetime.now().isoformat()
                        }
                        st.session_state.current_chat_id = new_id
                        save_chat(user_id, new_id, st.session_state.chats[new_id])

                st.rerun()

    st.divider()
    st.markdown("### ➕ Add Documents")

    uploaded_files = st.file_uploader(
        "Upload policy documents",
        type=["txt", "pdf", "docx", "csv", "xlsx"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Index uploaded files", use_container_width=True):
        with st.spinner("Reading and indexing documents..."):
            added_count = 0
            skipped_duplicates = []

            existing_items = collection.get()
            already_indexed_files = set(
                doc_id.split("::")[0] for doc_id in existing_items["ids"]
            )

            for uploaded_file in uploaded_files:
                if uploaded_file.name in already_indexed_files:
                    skipped_duplicates.append(uploaded_file.name)
                    continue

                save_path = os.path.join("uploaded_docs", uploaded_file.name)
                os.makedirs("uploaded_docs", exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                ext = os.path.splitext(uploaded_file.name)[1].lower()
                if ext in READERS:
                    try:
                        content = READERS[ext](save_path)
                        if content.strip():
                            chunks = chunk_text(content)
                            chunk_ids = [f"{uploaded_file.name}::chunk{i}" for i in range(len(chunks))]
                            embeddings = embedder.encode(chunks).tolist()
                            collection.add(documents=chunks, embeddings=embeddings, ids=chunk_ids)
                            added_count += 1
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {e}")

        st.success(f"Indexed {added_count} document(s)!")
        if skipped_duplicates:
            st.info(f"Skipped (already indexed): {', '.join(skipped_duplicates)}")
        st.rerun()
    st.divider()
    st.markdown("### 🧠 Add a note")

    if "clear_note" not in st.session_state:
        st.session_state.clear_note = False
    if st.session_state.clear_note:
        st.session_state.note_input = ""
        st.session_state.clear_note = False

    note_text = st.text_area(
        "Information for the assistant to remember",
        placeholder="e.g. The office WiFi password is Guest2026",
        height=100,
        key="note_input"
    )
    if st.button("Save note", use_container_width=True):
        if not note_text.strip():
            st.warning("Please enter some text first.")
        else:
            add_manual_knowledge(note_text.strip(), submitted_by=user_id)
            st.success("Note added!")
            st.session_state.clear_note = True
            st.rerun()


# --- Main chat area ---

st.title("💬 Company Policy Assistant")
st.caption("Answers are grounded in company documents. Runs entirely locally — nothing leaves this machine.")

chat = current_chat()

for i, msg in enumerate(chat["messages"]):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg["content"])
    else:
        row_main, row_sources = st.columns([3, 1])
        with row_main:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
        if msg.get("chunks"):
            render_sources(row_sources, msg["chunks"], msg["sources"], msg["content"], key_prefix=f"hist{i}")

user_input = st.chat_input("Ask a question about company policy...")

if user_input:
    chat["messages"].append({"role": "user", "content": user_input})

    if chat["title"] == "New chat":
        chat["title"] = user_input[:40]

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    row_main, row_sources = st.columns([3, 1])

    with row_main:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching documents..."):
                chunks, ids = retrieve_context_cached(user_input)

            if not chunks:
                answer = "I don't have any information relevant to that question in the documents I have access to."
                st.markdown(answer)
            else:
                answer = st.write_stream(ask_llm_stream(user_input, chunks))

                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("👍", key=f"up_{len(chat['messages'])}"):
                        log_feedback(user_input, answer, "up")
                        st.toast("Thanks for the feedback!")
                with col2:
                    if st.button("👎", key=f"down_{len(chat['messages'])}"):
                        log_feedback(user_input, answer, "down")
                        st.toast("Thanks — I'll note that.")

    if chunks:
        render_sources(row_sources, chunks, ids, answer, key_prefix="latest")

    log_interaction(user_input, ids if chunks else [], answer)
    chat["messages"].append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks if chunks else [],
        "sources": ids if chunks else []
    })

    persist_current_chat()