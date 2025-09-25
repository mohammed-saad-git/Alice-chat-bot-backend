# knowledge.py
"""
Knowledge module for Alice (user_db removed).
- user_db import and ingestion removed to prevent accidental leakage.
- Structured handlers (class, teacher), data.txt ingestion, vector DB, and Gemini fallback remain.
- Optional website scraper hook retained if you want on-demand scraping.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os
from dotenv import load_dotenv
import json
import re

# Optional website scraper integration (if you have website_scraper.py)
try:
    from website_scraper import summarize_website
except Exception:
    summarize_website = None

# ---------------------------
# Load API key
# ---------------------------
load_dotenv()

# ---------------------------
# Gemini model switcher
# ---------------------------
def get_llm(mode: str = "flash"):
    """Return Gemini LLM instance depending on mode."""
    model_name = "gemini-1.5-flash-latest" if mode == "flash" else "gemini-1.5-pro-latest"
    print(f"[INFO] Using Gemini model: {model_name}")
    return ChatGoogleGenerativeAI(model=model_name, convert_system_message_to_human=True)

# ---------------------------
# Embeddings
# ---------------------------
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("[OK] Embeddings initialized")
except Exception as e:
    print(f"[ERROR] Could not initialize embeddings: {e}")
    embeddings = None

# ---------------------------
# Teachers Data
# ---------------------------
teachers_data = {
    "headmistress": "name",
    "vice principal": "name",
    "physics": "name",
    "mathematics": "name",
    "computer": "nameh",
    "english": "name",
    "hindi": "name",
    "chemistry": "name",
    "computer science mentor": " name ",
    "computer science": "name",  # explicit mapping for common phrasing
}

# ---------------------------
# Class 2B structured data (all students)
# ---------------------------
class_data = {
    Your class data}
# ---------------------------
# Prompt template
# ---------------------------
prompt_template = """You are Alice, a helpful assistant.
Use the following context to answer the question. 
If the answer isn't in the context, just say:
"I don't have that specific information in my school database."

CONTEXT:
{context}

Question: {question}

Helpful Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ---------------------------
# Custom origin reply
# ---------------------------
CUSTOM_ORIGIN_REPLY = (
    "I was made as a clever AI project for the Science Fest at CMR National PU College. "
    "Saad built me to demonstrate practical AI applications for students and staff."
)

ORIGIN_PATTERNS = [
    r"why (were you|were you made|were you created)",
    r"who made you",
    r"who created you",
    r"why were you made",
    r"who built you",
    r"what is your purpose",
    r"why (are you here|are you here for)",
]
_origin_regexes = [re.compile(p, flags=re.I) for p in ORIGIN_PATTERNS]

def looks_like_origin_question(text: str) -> bool:
    if not text:
        return False
    for rx in _origin_regexes:
        if rx.search(text):
            return True
    return False

# ---------------------------
# Initialize Vector DB
# ---------------------------
def initialize_knowledge_base():
    """Load or create Chroma DB from data.txt + class data + teachers only.
    NOTE: no user personal data is ever added to this DB.
    """
    if not os.path.exists("./chroma_db"):
        print("[INFO] Creating new knowledge base...")
        documents = []

        # Load data.txt (user-provided / scraped content)
        if os.path.exists("data.txt"):
            try:
                loader = TextLoader("data.txt", encoding="utf-8")
                documents.extend(loader.load())
                print("[OK] Loaded data.txt")
            except Exception as e:
                print(f"[ERROR] Failed to load data.txt: {e}")

        # Add class 2B structured data
        class_text = f"Class 2B Teacher: {class_data['classTeacher']}\nStudents:\n"
        for s in class_data["students"]:
            class_text += f"- {s['full_name']} ({s['roll_no']}), {s['combination']}, {s['ii_language']}\n"
        documents.append(Document(page_content=class_text, metadata={"source": "class_2b"}))

        # Add teachers data
        teachers_text = "Teachers Information:\n"
        for subject, teacher in teachers_data.items():
            teachers_text += f"{subject.capitalize()}: {teacher}\n"
        documents.append(Document(page_content=teachers_text, metadata={"source": "teachers"}))

        # Build DB
        if documents and embeddings:
            try:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                texts = splitter.split_documents(documents)
                db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
                db.persist()
                print("[OK] Knowledge base created")
                return db
            except Exception as e:
                print(f"[ERROR] Creating vector DB failed: {e}")
                return None
        return None
    else:
        print("[INFO] Loading existing knowledge base...")
        try:
            return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        except Exception as e:
            print(f"[ERROR] Loading chroma DB failed: {e}")
            return None

# build initial vector_db (module global)
try:
    vector_db = initialize_knowledge_base()
    if vector_db:
        print("[OK] Vector DB ready")
    else:
        print("[WARNING] Vector DB not available")
except Exception as e:
    print(f"[ERROR] Knowledge base init failed: {e}")
    vector_db = None

# ---------------------------
# Helpers for ingestion & searching
# ---------------------------
def _strip_html(html: str) -> str:
    text = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?>.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def ingest_text(source: str, text: str):
    """
    Append text to data.txt and reinitialize the vector DB so new content is searchable.
    Returns True on success.
    """
    try:
        with open("data.txt", "a", encoding="utf-8") as fh:
            fh.write(f"\n\n# SOURCE: {source}\n")
            fh.write(text)
        global vector_db
        print(f"[INGEST] Rebuilding knowledge base after ingesting {source}...")
        vector_db = initialize_knowledge_base()
        return True
    except Exception as e:
        print(f"[ERROR] ingest_text failed: {e}")
        return False

def ingest_file(filepath: str, source: str = None):
    try:
        if not source:
            source = os.path.basename(filepath)
        ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
        text = ""
        if ext == "pdf":
            try:
                import fitz
                with fitz.open(filepath) as pdf:
                    pages = []
                    for page in pdf:
                        pages.append(page.get_text("text"))
                    text = "\n\n".join(pages)
            except Exception as e:
                print(f"[WARN] PDF extraction failed: {e}. Trying to read as binary text.")
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                except Exception as ee:
                    print(f"[ERROR] Could not read file: {ee}")
                    return False
        elif ext in ("html", "htm"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                html = fh.read()
            text = _strip_html(html)
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()

        if not text.strip():
            print("[INGEST] no text extracted from file")
            return False

        return ingest_text(source, text)
    except Exception as e:
        print(f"[ERROR] ingest_file failed: {e}")
        return False

def quick_search_local(query: str, top_k: int = 3):
    results = []
    q = query.lower().strip()
    try:
        if os.path.exists("data.txt"):
            with open("data.txt", "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            if q in content.lower():
                idx = content.lower().find(q)
                start = max(0, idx - 200)
                end = min(len(content), idx + 400)
                snippet = content[start:end].strip()
                results.append({"source": "data.txt", "text": snippet, "score": 0.8})
    except Exception:
        pass

    if vector_db:
        try:
            retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.get_relevant_documents(query)
            for d in docs:
                results.append({"source": d.metadata.get("source", "vector"), "text": d.page_content[:1000], "score": 0.65})
        except Exception as e:
            print(f"[WARN] quick_search_local vector lookup failed: {e}")

    return results

# ---------------------------
# Structured handlers
# ---------------------------
def handle_class_question(question: str):
    q = question.lower()
    if "teacher" in q and "2b" in q:
        return f"The class teacher for 2B is {class_data['classTeacher']}."
    for s in class_data["students"]:
        if s["full_name"].lower() in q:
            return f"{s['full_name']} → Roll {s['roll_no']}, {s['combination']}, {s['ii_language']}"
    return None

def handle_teacher_question(question: str):
    q = question.lower()
    # direct subject -> teacher map
    for subject, teacher in teachers_data.items():
        if subject in q and ("teacher" in q or "teach" in q or "who" in q):
            if subject in ("vice principal", "headmistress"):
                return f"{teacher} is the {subject}."
            return f"{teacher} teaches {subject.capitalize()}."
        if teacher.lower() in q:
            return f"{teacher} teaches {subject.capitalize()}."
    # also handle common aliases
    aliases = {
        "computer": ["computer", "computer science", "cs", "computer science mentor", "computer science teacher"],
        "physics": ["physics", "physic", "phys"],
        "mathematics": ["math", "mathematics", "maths"],
    }
    for canonical, keys in aliases.items():
        for k in keys:
            if k in q:
                mapped = teachers_data.get(canonical) or teachers_data.get("computer")
                if mapped:
                    return f"{mapped} teaches {canonical.capitalize()}."
    return None

def handle_data_txt_question(question: str):
    try:
        if os.path.exists("data.txt"):
            with open("data.txt", "r", encoding="utf-8") as f:
                content = f.read().lower()
            q = question.lower()
            if "your name" in q or re.search(r"\b(who are you|what('?s| is)? your name)\b", q):
                return "My name is Alice."
            if re.search(r"\b(who made you|who created you|who built you)\b", q):
                return CUSTOM_ORIGIN_REPLY
            if "vinayashree" in content and "teacher" in q:
                return "Vinayashree Bhat teaches Chemistry."
    except Exception as e:
        print(f"[ERROR] Reading data.txt: {e}")
    return None

# ---------------------------
# Main ask_question
# ---------------------------
def ask_question(question: str, mode: str = "flash"):
    """
    Main entry: answer a question using:
      1) assistant identity / origin
      2) structured handlers (class, teachers)
      3) quick local search (data.txt + vector DB)
      4) vector DB RetrievalQA (if available)
      5) Gemini fallback
    Returns: (answer_text, used_local_bool)
    """
    print(f"\n[INFO] Processing question: {question}")

    if not question or not question.strip():
        return "Please ask a question.", False

    q = question.strip()
    q_lower = q.lower()

    # 0) small talk
    #if re.match(r"^(hi|hello|hey|hiya|how are you|how's it going|good morning|good afternoon|good evening)\b", q_lower):
     #return "I'm doing great — thanks for asking! How can I help you today?", False

    # 1) assistant identity requests -> always return Alice
    if re.search(r"\b(what('?s| is)? your name|who are you|what are you called)\b", q_lower):
        return "My name is Alice.", True

    # 2) origin/purpose questions -> always return custom origin
    if looks_like_origin_question(q):
        return CUSTOM_ORIGIN_REPLY, True

    # 3) structured local checks (fast)
    ans = handle_class_question(q)
    if ans:
        return ans, True

    ans = handle_teacher_question(q)
    if ans:
        return ans, True

    ans = handle_data_txt_question(q)
    if ans:
        return ans, True

    # 4) quick local search (data.txt + vector DB)
    local_matches = []
    try:
        local_matches = quick_search_local(q, top_k=5)
    except Exception as e:
        print(f"[WARN] quick_search_local failed: {e}")
        local_matches = []

    college_keywords = ["cmr", "cmr pu", "cmr national", "cmr college", "cmr npuc", "cmr group"]
    if any(kw in q_lower for kw in college_keywords):
        if local_matches:
            top = local_matches[0]
            if len(top.get("text", "")) > 600:
                try:
                    llm = get_llm(mode)
                    prompt_text = PROMPT.template.replace("{context}", top.get("text", "")).replace("{question}", q)
                    resp = llm.invoke(prompt_text)
                    content = getattr(resp, "content", None) or str(resp)
                    return content, True
                except Exception as e:
                    print(f"[WARN] LLM summarization failed: {e}")
                    return f"From {top['source']}:\n\n{top['text']}", True
            return f"From {top['source']}:\n\n{top['text']}", True

        # optional on-demand website summarization (if website_scraper is available)
        if summarize_website:
            try:
                pages = [
                    "https://npuc.cmr.ac.in/",
                    "https://www.cmr.edu.in/history/",
                    "https://www.cmr.edu.in/cmruprograms/",
                    "https://www.cmr.edu.in/leadership/",
                ]
                combined = []
                for u in pages:
                    s = summarize_website(u)
                    if s and not s.lower().startswith("error"):
                        combined.append(f"Summary of {u}:\n{s}")
                if combined:
                    big = "\n\n".join(combined)
                    llm = get_llm(mode)
                    prompt_text = PROMPT.template.replace("{context}", big).replace("{question}", q)
                    resp = llm.invoke(prompt_text)
                    content = getattr(resp, "content", None) or str(resp)
                    return content, True
            except Exception as e:
                print(f"[WARN] website scraping fallback failed: {e}")

        fallback = (
            "CMR National PU College is part of the CMR Group of Institutions in Bangalore. "
            "It focuses on pre-university education and student development. "
            "If you want, I can ingest the official pages so I can answer in more detail."
        )
        return fallback, True

    # If there are confident local matches, prefer them (with summarization if long)
    if local_matches:
        snippets = []
        total_len = 0
        for m in local_matches:
            src = m.get("source", "local")
            txt = m.get("text", "")
            snippets.append(f"From {src}: {txt}")
            total_len += len(txt)
            if total_len > 3000:
                break
        combined_snippets = "\n\n".join(snippets)
        if len(combined_snippets) < 800:
            return combined_snippets, True
        try:
            llm = get_llm(mode)
            prompt_text = PROMPT.template.replace("{context}", combined_snippets).replace("{question}", q)
            resp = llm.invoke(prompt_text)
            content = getattr(resp, "content", None) or str(resp)
            return content, True
        except Exception as e:
            print(f"[WARN] LLM-with-local-context failed: {e}")
            return combined_snippets, True

    # 5) Vector DB (RetrievalQA) if available
    if vector_db:
        try:
            print("[INFO] Trying RetrievalQA over vector DB")
            qa_chain = RetrievalQA.from_chain_type(
                llm=get_llm(mode),
                chain_type="stuff",
                retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT},
            )
            result = qa_chain({"query": q})
            maybe = result.get("result") if isinstance(result, dict) else getattr(result, "answer", None) or str(result)
            if maybe and "i don't have" not in maybe.lower():
                print("[SOURCE] Answered from vector DB")
                return maybe, True
        except Exception as e:
            print(f"[WARN] Vector DB RetrievalQA failed: {e}")

    # 6) Gemini fallback (final)
    try:
        print("[INFO] Falling back to Gemini LLM")
        llm = get_llm(mode)
        resp = llm.invoke(q)
        content = getattr(resp, "content", None) or str(resp)
        return content, False
    except Exception as e:
        print(f"[ERROR] Gemini failed: {e}")

    return "Sorry, I couldn't find an answer for that.", False

