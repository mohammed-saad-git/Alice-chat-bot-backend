# api_server.py
import os
import re
from typing import Dict, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# Local project modules - must exist in your project
from user_data import user_db      # expected: get_user_data(user_id[, key]) & set_user_data(user_id, key, value)
from website_scraper import summarize_website
from knowledge import ask_question

# Configure tesseract path if needed (adjust on your machine)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {"pdf", "png", "jpg", "jpeg"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ---------- OCR helpers ----------
def extract_text_from_pdf(filepath: str) -> str:
    txt = ""
    with fitz.open(filepath) as pdf:
        for page in pdf:
            txt += page.get_text("text") + "\n"
    return txt.strip()


def extract_text_from_image(filepath: str) -> str:
    img = Image.open(filepath)
    text = pytesseract.image_to_string(img, lang="eng")
    return text.strip()


# ---------- health ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ---------- user memory endpoints ----------
@app.route("/api/user/get", methods=["GET"])
def api_get_user():
    user_id = request.args.get("user_id", "local_user")
    key = request.args.get("key")
    if key:
        val = user_db.get_user_data(user_id, key)
        return jsonify({"key": key, "value": val}), 200
    else:
        data = user_db.get_user_data(user_id)
        if data is None:
            data = {}
        return jsonify({"data": data}), 200


@app.route("/api/user/set", methods=["POST"])
def api_set_user():
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("user_id", "local_user")
    key = payload.get("key")
    value = payload.get("value")
    if not key:
        return jsonify({"error": "key required"}), 400
    user_db.set_user_data(user_id, key, value)
    return jsonify({"status": "ok", "key": key, "value": value}), 200


@app.route("/api/user/list", methods=["GET"])
def api_user_list():
    user_id = request.args.get("user_id", "local_user")
    data = user_db.get_user_data(user_id) or {}
    keys = list(data.keys()) if isinstance(data, dict) else []
    return jsonify({"keys": keys, "data": data}), 200


# ---------- website scraping ----------
@app.route("/api/scrape", methods=["POST"])
def api_scrape():
    data = request.get_json(silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "url required"}), 400
    try:
        summary = summarize_website(url)
        return jsonify({"summary": summary}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- upload (OCR + ask) ----------
@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        ext = filename.rsplit(".", 1)[1].lower()
        extracted_text = ""
        try:
            if ext == "pdf":
                extracted_text = extract_text_from_pdf(filepath)
            else:
                extracted_text = extract_text_from_image(filepath)
        except Exception as e:
            return jsonify({"error": f"OCR failed: {str(e)}"}), 500

        if not extracted_text:
            return jsonify({"error": "No readable text found in file"}), 400

        try:
            result = ask_question(extracted_text)
            if isinstance(result, tuple) and len(result) == 2:
                answer, conf = result
            else:
                answer = result
            return jsonify({"response": answer, "extracted_text": extracted_text[:2000], "content": answer}), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return jsonify({"error": "File type not allowed"}), 400


# ---------- Memory parsing & lookup helpers ----------
def _split_chunks(s: str):
    s2 = re.sub(r",\s*", " and ", s)
    parts = re.split(r"\s+and\s+|\s*&\s*", s2, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def parse_remember_statements(text: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    chunks = _split_chunks(text)
    for chunk in chunks:
        c = chunk.strip()

        # name patterns
        m = re.search(r"(?:my\s+)?name\s+(?:is|:)\s*(.+)$", c, flags=re.I)
        if m:
            facts["name"] = m.group(1).strip().rstrip(".")
            continue

        # likes / love / enjoy
        m = re.search(r"\bI\s+(?:like(?:\s+to)?|love|enjoy)\s+(.+)$", c, flags=re.I)
        if m:
            facts["likes"] = m.group(1).strip().rstrip(".")
            continue

        # favorite / favourite
        m = re.search(r"(?:my\s+)?(?:favorite|favourite)(?:\s+(?:color|colour|food|movie|thing))?\s*(?:is|:)\s*(.+)$", c, flags=re.I)
        if m:
            facts["favorite"] = m.group(1).strip().rstrip(".")
            continue

        # generic: "<key> is <value>"
        m = re.search(r"(?:remember(?:\s+that)?\s+)?([a-z0-9 _'-]+?)\s*(?:is|:|=)\s*(.+)$", c, flags=re.I)
        if m:
            key_raw = m.group(1).strip()
            val = m.group(2).strip().rstrip(".")
            key = re.sub(r"^my\s+", "", key_raw, flags=re.I).lower().replace(" ", "_")
            if key and val:
                if key not in facts:
                    facts[key] = val
            continue

        continue

    return facts


def find_memory_answer(message: str, memory: Dict[str, str]) -> Optional[str]:
    """
    Return a short answer from stored user memory for queries like:
      - "what is my name"
      - "do you remember my favorite food"
      - "what does <name> like"

    BUT: explicitly **do not** answer assistant-identity questions (e.g. "name", "your name",
    "who are you", "what is your name") from user memory — those should be handled by the
    knowledge/LLM layer so the assistant always returns its configured identity ("Alice").
    """
    s = (message or "").strip().lower()

    # --- Guard: never let memory override assistant identity ---
    # If the user asks for the assistant's name or identity, return None so the request
    # falls through to knowledge.ask_question (which returns "My name is Alice.")
    if not s:
        return None
    # exact single-word "name" or short queries about "your name" / "who are you" -> block memory
    if s == "name" or "your name" in s or re.search(r"\b(who are you|what('?s| is)? your name)\b", s):
        return None

    # --- existing memory lookup logic (unchanged) ---
    # 1) "what is my <key>"
    m = re.search(r"what (?:is|are) my\s+([a-z0-9 _-]+)\??", s, flags=re.I)
    if m:
        key_raw = m.group(1).strip()
        key = key_raw.lower().replace(" ", "_")
        key_map = {"name": "name", "favorite": "favorite", "favourite": "favorite", "likes": "likes", "like": "likes"}
        key = key_map.get(key, key)
        val = memory.get(key)
        if val:
            label = key.replace("_", " ")
            return f"{label.capitalize()}: {val}."

    # 2) "do you remember my <key>"
    m = re.search(r"do you remember (?:my\s+)?([a-z0-9 _-]+)\??", s, flags=re.I)
    if m:
        key_raw = m.group(1).strip()
        key = key_raw.lower().replace(" ", "_")
        val = memory.get(key)
        if val:
            return f"Yes — {key_raw.strip()} = {val}."

    # 3) "what does <name> like"
    m = re.search(r"what does\s+([a-z0-9 _-]+)\s+like(?:\s+to\s+eat)?\b", s, flags=re.I)
    if m:
        asked_name = m.group(1).strip().lower()
        stored_name = (memory.get("name") or "").lower()
        if stored_name and asked_name == stored_name:
            likes = memory.get("likes") or memory.get("favorite")
            if likes:
                return f"{memory.get('name')} likes {likes}."

    # generic key presence checks (keep these)
    for k, v in memory.items():
        pretty = k.lower().replace("_", " ")
        if pretty and pretty in s:
            return f"{k.replace('_', ' ').capitalize()}: {v}."

    return None



# ---------- chat endpoint (accepts JSON or multipart/form-data with optional file) ----------
@app.route("/api/chat", methods=["POST"])
def chat():
    # Try JSON first (application/json)
    json_payload = request.get_json(silent=True)
    form_payload = request.form if request.form else None
    files = request.files

    # Determine message and file (if any)
    message = ""
    if json_payload:
        message = (json_payload.get("message") or json_payload.get("content") or "").strip()
        mode = json_payload.get("mode", "flash")
        user_id = json_payload.get("user_id", "local_user")
    elif form_payload:
        # form-data case
        message = (form_payload.get("message") or form_payload.get("content") or "").strip()
        mode = form_payload.get("mode", "flash")
        user_id = form_payload.get("user_id", "local_user")
    else:
        # fallback: try args
        message = (request.args.get("message") or request.args.get("content") or "").strip()
        mode = request.args.get("mode", "flash")
        user_id = request.args.get("user_id", "local_user")

    # if file is present (multipart), handle OCR path
    file = files.get("file") if files and "file" in files else None

    if not message and not file:
        return jsonify({"response": "⚠️ No message or file provided", "content": "⚠️ No message or file provided"}), 400

    # Load memory safely
    try:
        memory = user_db.get_user_data(user_id) or {}
    except Exception:
        memory = {}

    # If file provided -> save + OCR + ask_question over extracted text
    if file:
        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            return jsonify({"response": "File type not allowed", "content": "File type not allowed"}), 400
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        ext = filename.rsplit(".", 1)[1].lower()
        extracted_text = ""
        try:
            if ext == "pdf":
                extracted_text = extract_text_from_pdf(filepath)
            else:
                extracted_text = extract_text_from_image(filepath)
        except Exception as e:
            return jsonify({"response": f"OCR failed: {str(e)}", "content": f"OCR failed: {str(e)}"}), 500

        if not extracted_text:
            return jsonify({"response": "No readable text found in file", "content": "No readable text found in file"}), 400

        # Optionally combine with message (if user provided some extra text)
        query_text = (message + "\n\n" + extracted_text).strip() if message else extracted_text

        try:
            result = ask_question(query_text, mode)
            if isinstance(result, tuple) and len(result) == 2:
                answer, conf = result
            else:
                answer = result
            # return extracted_text summary in response (but limit size)
            return jsonify({"response": answer, "content": answer, "extracted_text": extracted_text[:4000]}), 200
        except Exception as e:
            return jsonify({"response": f"⚠️ LLM error: {str(e)}", "content": f"⚠️ LLM error: {str(e)}"}), 500

    # ---------- If we reach here: no file, handle text message ----------
    text_lower = message.lower()

    # 1) Remember statements
    if "remember" in text_lower:
        facts = parse_remember_statements(message)
        stored = []
        for k, v in facts.items():
            try:
                user_db.set_user_data(user_id, k, v)
                stored.append(f"{k} = {v}")
            except Exception:
                pass
        if stored:
            resp = f"Okay, I remember {', '.join(stored)}."
            return jsonify({"response": resp, "content": resp}), 200
        else:
            resp = "Okay — noted. If I didn't get it, try 'remember my name is Saad' style."
            return jsonify({"response": resp, "content": resp}), 200

    # 2) Memory-first answers
    mem_ans = find_memory_answer(message, memory if isinstance(memory, dict) else {})
    if mem_ans:
        return jsonify({"response": mem_ans, "content": mem_ans}), 200

    # 3) Fall back to LLM/knowledge
    try:
        result = ask_question(message, mode)
        if isinstance(result, tuple) and len(result) == 2:
            answer, conf = result
        else:
            answer = result
        return jsonify({"response": answer, "content": answer}), 200
    except Exception as e:
        return jsonify({"response": f"⚠️ LLM error: {str(e)}", "content": f"⚠️ LLM error: {str(e)}"}), 500


# ---------- run ----------
if __name__ == "__main__":
    is_dev = os.environ.get("ELECTRON_DEV", "0") == "1"
    app.run(port=3001, host="127.0.0.1", debug=is_dev)
