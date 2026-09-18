import os

import chromadb

from sentence_transformers import SentenceTransformer

# --- Format-specific readers ---

def read_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(filepath):
    """Extracts text from a PDF using layout-aware parsing, falling back
    to OCR if the PDF turns out to be a scanned image with no real text."""
    import pdfplumber

    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        text = ""

    # If normal extraction produced almost nothing, this is very likely
    # a scanned/image-based PDF — fall back to OCR
    if len(text.strip()) < 50:
        text = _ocr_pdf(filepath)

    return text


def _ocr_pdf(filepath):
    """Reads a PDF by converting each page to an image and running OCR on it.
    Used only when normal text extraction fails (scanned documents)."""
    from pdf2image import convert_from_path
    import pytesseract

    text = ""
    try:
        images = convert_from_path(filepath)
        for image in images:
            text += pytesseract.image_to_string(image) + "\n"
    except Exception as e:
        print(f"OCR failed for {filepath}: {e}")
        return ""

    return text

def read_docx(filepath):
    import docx
    doc = docx.Document(filepath)
    return "\n".join(para.text for para in doc.paragraphs)

def read_csv(filepath):
    import pandas as pd
    df = pd.read_csv(filepath)
    # Convert each row into a readable sentence rather than raw table dump
    lines = []
    for _, row in df.iterrows():
        line = ", ".join(f"{col}: {row[col]}" for col in df.columns)
        lines.append(line)
    return "\n".join(lines)

def read_excel(filepath):
    import pandas as pd
    df = pd.read_excel(filepath)
    lines = []
    for _, row in df.iterrows():
        line = ", ".join(f"{col}: {row[col]}" for col in df.columns)
        lines.append(line)
    return "\n".join(lines)

# Maps file extension -> reader function
READERS = {
    ".txt": read_txt,
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".csv": read_csv,
    ".xlsx": read_excel,
    ".xls": read_excel,
}

def load_documents(folders):
    """Reads every supported file from multiple folders."""

    documents = []
    doc_ids = []
    skipped = []

    # Allow both a single folder and multiple folders
    if isinstance(folders, str):
        folders = [folders]

    for folder in folders:

        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        print(f"\nReading folder: {os.path.abspath(folder)}")

        for filename in os.listdir(folder):

            filepath = os.path.join(folder, filename)

            # Skip directories
            if not os.path.isfile(filepath):
                continue

            ext = os.path.splitext(filename)[1].lower()

            if ext not in READERS:
                skipped.append(
                    f"{folder}/{filename} (unsupported type)"
                )
                continue

            try:
                content = READERS[ext](filepath)

                # DEBUG
                print("\n==============================")
                print(f"FILE: {filename}")
                print(f"FOLDER: {folder}")
                print(f"TYPE: {ext}")
                print(f"CONTENT LENGTH: {len(content)}")
                print("PREVIEW:")
                print(repr(content[:500]))
                print("==============================\n")

                if content and content.strip():

                    documents.append(content)

                    # Important: include folder in ID
                    # Otherwise same filename in both folders can conflict
                    unique_id = f"{folder}/{filename}"

                    doc_ids.append(unique_id)

                else:
                    skipped.append(
                        f"{folder}/{filename} (empty content)"
                    )

            except Exception as e:

                skipped.append(
                    f"{folder}/{filename} (error: {e})"
                )

    return documents, doc_ids, skipped


def chunk_text(text, chunk_size=500, overlap=50):
    """Splits long text into overlapping chunks so retrieval finds specific
    passages rather than whole (possibly very long) documents."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index(
    docs_folders=None,
    db_path="./chroma_db",
    collection_name="policies",
    rebuild=False
):

    if docs_folders is None:
        docs_folders = [
            "dummy_docs",
            "uploaded_docs"
        ]
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    client = chromadb.PersistentClient(path=db_path)

    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name)

    # Find out which source files are already indexed, so we can skip them
    existing_items = collection.get()
    already_indexed_files = set(
        doc_id.split("::")[0] for doc_id in existing_items["ids"]
    )

    print(f"Reading documents from: {docs_folders}")

    documents, doc_ids, skipped = load_documents(
        docs_folders
    )

    # Only process files we haven't indexed yet, unless rebuild=True
    new_documents = []
    new_doc_ids = []
    for doc_text, doc_id in zip(documents, doc_ids):
        if rebuild or doc_id not in already_indexed_files:
            new_documents.append(doc_text)
            new_doc_ids.append(doc_id)

    print(f"Found {len(documents)} file(s) total, {len(new_documents)} new/changed")
    if skipped:
        print(f"Skipped: {skipped}")

    if not new_documents:
        print("Nothing new to index.")
        return

    all_chunks = []
    all_chunk_ids = []
    for doc_text, doc_id in zip(new_documents, new_doc_ids):
        chunks = chunk_text(doc_text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_chunk_ids.append(f"{doc_id}::chunk{i}")

    print(f"Split into {len(all_chunks)} new chunk(s)")

    embeddings = embedder.encode(all_chunks).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_chunk_ids
    )

    print("Documents embedded and stored successfully!")
    print(f"Collection now has {collection.count()} chunks total.")


if __name__ == "__main__":
    build_index(rebuild=True) # incremental by default
    # build_index(rebuild=True)  # use this instead for a full rebuild
