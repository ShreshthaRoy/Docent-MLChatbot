# Company Policy Assistant

A confidential-data chatbot that answers questions using your organization's documents, powered by 
retrieval-augmented generation (RAG). Runs entirely locally — no document content is ever sent to an 
external API.

## Architecture

- **Ollama** — runs the language model (Llama 3.2) locally, offline
- **ChromaDB** — stores document embeddings for semantic search
- **sentence-transformers** — converts text into embeddings
- **Streamlit** — the chat interface

## Setup

### 1. Install Python
Python 3.10–3.13 recommended.

### 2. Create and activate a virtual environment
```bash
python3 -m venv agents_env
source agents_env/bin/activate      # Mac/Linux
agents_env\Scripts\activate         # Windows
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh    # Mac/Linux
```
Windows: download the installer from ollama.com.

### 5. Download the language model
```bash
ollama pull llama3.2
```

### 6. Start Ollama
```bash
ollama serve
```
Leave running. If it says "address already in use," it's already running — that's fine.

### 7. Run the app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Enter any name/email when prompted, then ask a question.

## Quick start (after first-time setup)
```bash
./start.sh
```

## Project structure

.
├── app.py # Main Streamlit application
├── chat_store.py # Per-user chat history persistence
├── build_index_multiformat.py # Document ingestion + indexing (txt/pdf/docx/csv/xlsx, OCR for scanned PDFs)
├── logger.py # Interaction logging
├── eval_harness.py # Automated accuracy testing
├── eval_questions.csv # Test question set
├── dummy_docs/ # Sample placeholder documents
├── chroma_db/ # Vector database (included, pre-built)
├── chat_history/ # Saved chats (created on first run, per user)
├── requirements.txt
└── start.sh



## Adding documents
- Through the app: sidebar → "Add Documents" → upload, indexed immediately
- Via script: drop files into `dummy_docs/`, run `python3 build_index_multiformat.py` (only processes new files)

## Running evaluations
```bash
python3 eval_harness.py
```

## Notes
- Runs fully offline once set up.
- The name/email prompt on login is for keeping chats separated between users — it is not real authentication.
- `chroma_db/`, `chat_history/`, `uploaded_docs/`, and log files may contain real content once used with actual data — do not commit these publicly. See `.gitignore`.
