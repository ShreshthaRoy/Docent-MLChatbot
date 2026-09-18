import csv
import os
from datetime import datetime

LOG_FILE = "chat_log.csv"

def log_interaction(question, retrieved_sources, answer):
    """Appends one row per interaction to a local CSV log file."""
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "retrieved_sources", "answer"])
        writer.writerow([
            datetime.now().isoformat(),
            question,
            "; ".join(retrieved_sources),
            answer
        ])