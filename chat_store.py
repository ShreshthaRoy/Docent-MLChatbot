import os
import json
import re

CHATS_DIR = "chat_history"


def _safe_user_folder(user_id):
    """Turns a name/email into a safe folder name."""
    safe = re.sub(r'[^a-zA-Z0-9_.@-]', '_', user_id.strip().lower())
    return safe or "unknown_user"


def _user_dir(user_id):
    path = os.path.join(CHATS_DIR, _safe_user_folder(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def load_all_chats(user_id):
    """Reads every saved chat belonging to this specific user."""
    chats = {}
    user_dir = _user_dir(user_id)
    for filename in os.listdir(user_dir):
        if filename.endswith(".json"):
            chat_id = filename[:-5]
            filepath = os.path.join(user_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    chats[chat_id] = json.load(f)
            except Exception:
                continue
    return chats


def save_chat(user_id, chat_id, chat_data):
    """Writes one chat to this user's own folder."""
    user_dir = _user_dir(user_id)
    filepath = os.path.join(user_dir, f"{chat_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=2)


def delete_chat(user_id, chat_id):
    user_dir = _user_dir(user_id)
    filepath = os.path.join(user_dir, f"{chat_id}.json")
    if os.path.isfile(filepath):
        os.remove(filepath)