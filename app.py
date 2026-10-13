"""
MLF Fisheries Regulatory Compliance AI
Streamlit UI inspired by the supplied MLF Chatbot reference image.

Architecture:
Streamlit -> LangGraph -> language/intent routing -> PGVector/BGE-M3
retrieval -> relevance grading -> optional query rewrite -> Groq -> answer/sources.

Configure DATABASE_URL, GROQ_API_KEY and optional settings in .env.
Self-registered accounts are standard users. Only the configured administrator
can upload/index documents.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, TypedDict

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_groq import ChatGroq
from datetime import datetime, timedelta
from langgraph.graph import END, StateGraph

from ingest import SUPPORTED_EXTENSIONS, load_file

load_dotenv()
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
COLLECTION = os.getenv("PGVECTOR_COLLECTION", "mlf_fisheries_regulations")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
TOP_K = max(1, int(os.getenv("TOP_K", "6")))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.48"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()


# ---------------------------------------------------------------------------
# Page configuration and visual styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MLF Chatbot | Ministry of Livestock and Fisheries",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root { --mlf-navy:#06284a; --mlf-blue:#0866c6; --mlf-pale:#eef5ff; }
[data-testid="stAppViewContainer"] { background:#f7f9fc; }
[data-testid="stHeader"] { background:rgba(255,255,255,.96); }
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#062746 0%,#07385a 72%,#062746 100%);
  border-right:1px solid #d8e3f0;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption { color:#eef6ff !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color:white !important; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.18); }
[data-testid="stSidebar"] [data-testid="stButton"] button {
  width:100%; text-align:left; min-height:42px; border-radius:9px;
  border:1px solid rgba(255,255,255,.19); background:rgba(255,255,255,.055);
  color:#f5f9ff; font-weight:550;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
  background:#1769c2; border-color:#4b9aff; color:white;
}
.mlf-brand { padding:10px 5px 20px; border-bottom:1px solid #31516e; margin-bottom:16px; }
.mlf-brand-row { display:flex; align-items:center; gap:11px; }
.mlf-crest { font-size:34px; line-height:1; }
.mlf-brand-title { color:white; font-size:15px; line-height:1.25; font-weight:750; }
.mlf-brand-sub { color:#c1d5e6; font-size:10px; line-height:1.4; margin-top:8px; }
.mlf-profile { display:flex; gap:10px; align-items:center; padding:12px 4px; margin-top:12px; }
.mlf-avatar { width:38px; height:38px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; background:#079b66; color:white; font-weight:800; flex-shrink:0; }
.mlf-profile-name { color:white; font-size:12px; font-weight:700; overflow-wrap:anywhere; }
.mlf-profile-role { color:#c1d5e6; font-size:10px; margin-top:3px; }
.mlf-topbar { background:white; border:1px solid #e3eaf3; border-radius:13px;
  padding:13px 18px; margin-bottom:14px; display:flex; align-items:center; gap:12px; }
.mlf-bot { width:42px; height:42px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; background:#0875e1; color:white; font-size:22px; flex-shrink:0; }
.mlf-app-title { font-size:20px; color:#132e4d; font-weight:800; line-height:1.2; }
.mlf-app-subtitle { font-size:11px; color:#5f7188; margin-top:4px; }
.mlf-chat-panel { background:#fff; border:1px solid #e5ebf3; border-radius:14px;
  min-height:60vh; padding:22px; box-shadow:0 2px 12px rgba(20,50,90,.035); }
.mlf-welcome { background:#edf4ff; border:1px solid #e0ebfa; border-radius:14px;
  padding:18px 20px; max-width:680px; color:#172d49; line-height:1.55; }
.mlf-welcome p { margin:5px 0; }
.mlf-shortcut { border-radius:11px; padding:14px 12px; border:1px solid #e3eaf4;
  min-height:78px; background:#fff; font-weight:700; font-size:12px; }
.mlf-shortcut-green { background:#edfff4; border-color:#d0f2dd; color:#117344; }
.mlf-shortcut-blue { background:#edf7ff; border-color:#d8eaff; color:#145ca5; }
.mlf-shortcut-gold { background:#fff8e9; border-color:#f5e7c6; color:#96610b; }
.mlf-shortcut-purple { background:#f6f0ff; border-color:#e9ddff; color:#6139bb; }
.mlf-login-card { background:white; border:1px solid #e0e8f2; border-radius:18px;
  padding:28px; box-shadow:0 10px 34px rgba(10,43,78,.08); }
div[data-testid="stChatMessage"] { border-radius:12px; }
[data-testid="stChatInput"] { border-color:#dbe4f0; }
footer { visibility:hidden; }
@media(max-width:700px) {
 .mlf-topbar { padding:10px; } .mlf-app-title {font-size:17px;}
 .mlf-chat-panel {padding:12px;} .mlf-welcome {padding:14px;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Authentication and registration
# ---------------------------------------------------------------------------
@st.cache_resource
def get_auth_engine():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing. Configure it in .env.")
    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url, pool_pre_ping=True, pool_recycle=300)


def initialize_user_database():
    with get_auth_engine().begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS mlf_user_accounts (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                full_name VARCHAR(150) NOT NULL,
                password_hash VARCHAR(128) NOT NULL,
                password_salt VARCHAR(64) NOT NULL,
                role VARCHAR(20) NOT NULL DEFAULT 'user'
                    CHECK (role IN ('user','admin')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))


def hash_password(password: str, salt: str | None = None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 600_000
    ).hex()
    return digest, salt


def verify_password(password: str, saved_hash: str, salt: str) -> bool:
    calculated, _ = hash_password(password, salt)
    return hmac.compare_digest(calculated, saved_hash)


def register_user(full_name: str, username: str, password: str):
    full_name, username = full_name.strip(), username.strip().lower()
    if len(full_name) < 2:
        return False, "Please enter your full name."
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{3,50}", username):
        return False, "Username must be 3–50 letters, numbers, dots, underscores or hyphens."
    if len(password) < 12:
        return False, "Password must contain at least 12 characters."
    try:
        digest, salt = hash_password(password)
        with get_auth_engine().begin() as conn:
            conn.execute(text("""
                INSERT INTO mlf_user_accounts
                (username, full_name, password_hash, password_salt, role)
                VALUES (:username,:full_name,:digest,:salt,'user')
            """), {"username": username, "full_name": full_name,
                   "digest": digest, "salt": salt})
        return True, "Account created. You can now sign in."
    except Exception as exc:
        print(f"Registration error: {exc}")
        if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
            return False, "That username is already registered."
        return False, "Unable to create account. Check the database connection."


def authenticate(username: str, password: str):
    username = username.strip().lower()
    if (ADMIN_USERNAME and ADMIN_PASSWORD
            and hmac.compare_digest(username, ADMIN_USERNAME)
            and hmac.compare_digest(password, ADMIN_PASSWORD)):
        return "admin", username
    try:
        with get_auth_engine().connect() as conn:
            account = conn.execute(text("""
                SELECT username, password_hash, password_salt, role
                FROM mlf_user_accounts WHERE username=:username
            """), {"username": username}).mappings().first()
        if account and verify_password(password, account["password_hash"], account["password_salt"]):
            return account["role"], account["username"]
    except Exception as exc:
        print(f"Authentication error: {exc}")
    return None, None


def render_auth():
    try:
        initialize_user_database()
    except Exception as exc:
        st.error("The authentication database is unavailable. Check DATABASE_URL.")
        st.caption(str(exc))
        st.stop()

    _, center, _ = st.columns([1, 1.05, 1])
    with center:
        st.markdown('<div style="text-align:center;padding:35px 0 12px">'
                    
                    '<h2 style="color:#12365a;margin-bottom:4px">THE UNITED REPUBLIC OF TANZANIA</h2>'
                    '<p style="color:#61748a">AI Assistant for Livestock and Fisheries Services</p>'
                    '</div>', unsafe_allow_html=True)
        
        if st.session_state.get("show_register", False):
            st.subheader("Create your account")
            st.caption("New accounts have standard-user access.")
            with st.form("register_form"):
                full_name = st.text_input("Full name", placeholder="Enter your full name")
                new_username = st.text_input("Username", placeholder="Choose a username")
                password = st.text_input("Password", type="password", placeholder="At least 12 characters")
                confirm = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if submitted:
                if password != confirm:
                    st.error("The passwords do not match.")
                else:
                    ok, message = register_user(full_name, new_username, password)
                    if ok:
                        st.session_state.show_register = False
                        st.session_state.auth_notice = message
                        st.rerun()
                    else:
                        st.error(message)
            if st.button("Already registered? Sign in", use_container_width=True):
                st.session_state.show_register = False
                st.rerun()
        else:
            
      
           
        
            st.subheader("Welcome!")
            st.caption("Sign in to access the Ministry knowledge assistant.")

            notice = st.session_state.pop("auth_notice", None)
            if notice:
                st.success(notice)

            with st.form("login_form"):
                username = st.text_input(
                    "Username",
                    placeholder="Enter your username",
                    key="login_username",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password",
                )
                submitted = st.form_submit_button(
                    "Sign in",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                role, saved_username = authenticate(username, password)

                if role:
                    st.session_state.authenticated = True
                    st.session_state.username = saved_username
                    st.session_state.role = role
                    st.session_state.messages = []
                    st.session_state.thread_id = str(time.time_ns())
                    st.session_state.active_page = "Home"

                    st.session_state.pop("show_register", None)
                    st.session_state.pop("auth_notice", None)
                    st.session_state.pop("pending_question", None)

                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            if st.button(
                "New here? Create an account",
                key="open_register_button",
                use_container_width=True,
            ):
                st.session_state.show_register = True
                st.rerun()

        


if not st.session_state.get("authenticated", False):
    render_auth()
    st.stop()


# ---------------------------------------------------------------------------
# Retrieval and agent workflow
# ---------------------------------------------------------------------------
def validate_config():
    missing = [name for name, value in
               (("DATABASE_URL", DATABASE_URL), ("GROQ_API_KEY", GROQ_API_KEY)) if not value]
    if missing:
        raise RuntimeError("Missing .env setting(s): " + ", ".join(missing))


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
    )


@st.cache_resource
def get_vectorstore():
    validate_config()
    return PGVector(
        embeddings=get_embeddings(), collection_name=COLLECTION,
        connection=DATABASE_URL, use_jsonb=True, create_extension=False,
    )


@st.cache_resource
def get_llm():
    validate_config()
    return ChatGroq(model=GROQ_MODEL, temperature=0.1, max_retries=2, api_key=GROQ_API_KEY)


class AgentState(TypedDict, total=False):
    question: str
    language: str
    intent: str
    rewritten_query: str
    documents: list[dict[str, Any]]
    relevance: float
    answer: str
    sources: list[dict[str, Any]]
    attempts: int
    route: str


def detect_language(state: AgentState) -> AgentState:
    q = state["question"].strip().lower()
    sw_words = {"naomba", "vibali", "uvuvi", "samaki", "mvuvi", "nyavu", "leseni",
                "marufuku", "msimu", "adhabu", "kanuni", "sheria", "kibali",
                "masharti", "chombo", "eneo", "fomu", "habari", "asante",
                "kwaheri", "tafadhali", "nini", "vipi", "lini", "wapi", "je"}
    tokens = set(re.findall(r"\b[\wÀ-ÿ]+\b", q))
    language = "sw" if tokens & sw_words else "en"
    regulatory = any(term in q for term in (
        "kibali", "vibali", "uvuvi", "samaki", "mvuvi", "kanuni", "sheria",
        "leseni", "permit", "license", "licence", "fishing", "regulation",
        "form", "fomu", "masharti", "utaratibu", "ada", "adhabu", "chombo cha uvuvi"
    ))
    casual = {"kwaheri", "bye", "goodbye", "see you", "tutaonana", "asante",
              "thanks", "thank you", "habari", "hello", "hi", "mambo", "karibu", "shikamoo"}
    is_casual = not regulatory and (q in casual or q.startswith(("asante ", "thanks ", "thank you ", "kwaheri ")))
    return {**state, "language": language, "intent": "casual" if is_casual else "regulatory"}


def route_after_language(state: AgentState):
    return "direct" if state.get("intent") == "casual" else "retrieve"


def direct_response(state: AgentState) -> AgentState:
    q, lang = state["question"].strip().lower(), state.get("language", "en")
    if any(x in q for x in ("asante", "thanks", "thank you")):
        answer = "Karibu!" if lang == "sw" else "You're welcome!"
    elif q in {"kwaheri", "bye", "goodbye", "see you", "tutaonana"}:
        answer = "Kwaheri! Karibu tena." if lang == "sw" else "You're welcome. Goodbye!"
    elif q in {"habari", "hello", "hi", "mambo", "shikamoo"}:
        answer = ("Habari! Naweza kukusaidia kuhusu masuala ya uvuvi, vibali na kanuni."
                  if lang == "sw" else "Hello! How can I help you with fisheries regulations?")
    else:
        answer = "Karibu! Naweza kukusaidia." if lang == "sw" else "Hello! How can I help?"
    return {**state, "answer": answer, "sources": [], "relevance": 0.0}


def requested_form(query: str):
    match = re.search(r"\bfomu\s+(?:(?:na|namba)\s*\.?\s*)?(\d+)\b", query.lower())
    return match.group(1) if match else None


def source_key(doc: Document):
    meta = doc.metadata or {}
    return str(meta.get("file_id") or meta.get("source") or meta.get("document") or "unknown")


def source_info(doc: Document):
    meta = doc.metadata or {}
    location = []
    for key, label in (("page", "page"), ("sheet", "sheet"), ("slide", "slide")):
        if meta.get(key):
            location.append(f"{label} {meta[key]}")
    return {"source": meta.get("source") or meta.get("document") or "Unknown",
            "location": ", ".join(location), "file_type": meta.get("file_type", "")}


def retrieve(state: AgentState) -> AgentState:
    query = state.get("rewritten_query") or state["question"]
    try:
        store = get_vectorstore()
        candidate_k = max(TOP_K * 5, 30)
        try:
            results = store.max_marginal_relevance_search_with_score(
                query, k=candidate_k, fetch_k=max(candidate_k * 2, 60), lambda_mult=0.65
            )
        except AttributeError:
            results = store.similarity_search_with_relevance_scores(query, k=candidate_k)
    except Exception as exc:
        print(f"Retrieval error: {exc}")
        return {**state, "documents": [], "sources": [], "relevance": 0.0}

    form_number = requested_form(query)
    terms = [t.strip(".,:;!?()[]{}\"'") for t in query.lower().split()]
    terms = [t for t in terms if len(t) >= 3 and t not in {
        "the", "and", "for", "what", "how", "are", "kwa", "katika", "hii", "hiyo", "je", "nini"
    }]
    ranked = []
    for rank, (doc, vector_score) in enumerate(results):
        content = (doc.page_content or "").strip()
        score = float(vector_score)
        if terms:
            score += sum(1 for term in terms if term in content.lower()) / len(terms) * 0.20
        if form_number:
            if re.search(rf"\bfomu\s+(?:(?:na|namba)\s*\.?\s*)?{re.escape(form_number)}\b", content.lower()):
                score += 0.50
            elif re.search(r"\bfomu\s+(?:(?:na|namba)\s*\.?\s*)?\d+\b", content.lower()):
                score -= 0.25
        ranked.append((score, float(vector_score), rank, doc))
    ranked.sort(key=lambda row: row[0], reverse=True)

    chosen, counts, seen = [], {}, set()
    for score, vector_score, rank, doc in ranked:
        key = source_key(doc)
        normalized = " ".join((doc.page_content or "").split())
        if counts.get(key, 0) >= 3 or normalized in seen:
            continue
        counts[key] = counts.get(key, 0) + 1
        seen.add(normalized)
        chosen.append((score, vector_score, doc))
        if len(chosen) >= TOP_K:
            break

    documents, sources = [], []
    for score, vector_score, doc in chosen:
        documents.append({"content": doc.page_content, "metadata": doc.metadata,
                          "score": score, "vector_score": vector_score})
        info = source_info(doc)
        info["score"] = score
        sources.append(info)
    best = chosen[0][0] if chosen else 0.0
    return {**state, "documents": documents, "sources": sources, "relevance": best}


def grade_documents(state: AgentState) -> AgentState:
    if not state.get("documents") or (
        state.get("relevance", 0.0) < MIN_RELEVANCE and state.get("attempts", 0) < 1
    ):
        return {**state, "route": "rewrite" if state.get("attempts", 0) < 1 else "generate"}
    return {**state, "route": "generate"}


def rewrite_query(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the question as one precise Tanzanian fisheries-regulation retrieval query. Preserve any form number. Return only the query."),
        ("human", "{question}")
    ])
    response = get_llm().invoke(prompt.format_messages(question=state["question"]))
    return {**state, "rewritten_query": str(response.content).strip(),
            "attempts": state.get("attempts", 0) + 1}


def generate(state: AgentState) -> AgentState:
    parts, seen = [], set()
    for item in state.get("documents", []):
        # The retrieval node intentionally serializes Documents as dictionaries.
        content = str(item.get("content", "")).strip() if isinstance(item, dict) else str(getattr(item, "page_content", "")).strip()
        if content and " ".join(content.split()) not in seen:
            seen.add(" ".join(content.split()))
            parts.append(content)
    context = "\n\n---\n\n".join(parts)[:9000]
    lang = "Swahili" if state.get("language") == "sw" else "English"
    if not context:
        answer = ("Taarifa hiyo haikupatikana katika nyaraka zilizopo kwenye kanzidata ya maarifa ya Wizara."
                  if lang == "Swahili" else
                  "The requested information was not found in the available Ministry knowledge base.")
        return {**state, "answer": answer}
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the MLF Fisheries Regulatory Compliance AI Assistant.
Answer ONLY from the supplied Ministry knowledge-base context. Never invent laws,
fees, penalties, dates, procedures, permits, licences, conditions or requirements.
If evidence is insufficient, say it was not found in the available Ministry knowledge base.
Answer the exact question concisely in {language}. Do not use outside sources."""),
        ("human", "Question: {question}\n\nMinistry Knowledge Base:\n{context}\n\nAnswer using only this context.")
    ])
    response = get_llm().invoke(prompt.format_messages(
        language=lang, question=state["question"], context=context
    ))
    answer = response.content
    if not isinstance(answer, str):
        answer = str(answer)
    return {**state, "answer": answer.strip()}


@st.cache_resource
def get_graph():
    builder = StateGraph(AgentState)
    for name, fn in [
        ("language", detect_language), ("direct_response", direct_response),
        ("retrieve", retrieve), ("grade", grade_documents),
        ("rewrite", rewrite_query), ("generate", generate)
    ]:
        builder.add_node(name, fn)
    builder.set_entry_point("language")
    builder.add_conditional_edges("language", route_after_language,
                                  {"direct": "direct_response", "retrieve": "retrieve"})
    builder.add_edge("direct_response", END)
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", lambda state: state["route"],
                                  {"rewrite": "rewrite", "generate": "generate"})
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    return builder.compile()


def require_admin():
    if st.session_state.get("role") == "admin":
        return True
    st.error("Access denied. Only administrators can upload documents.")
    return False


def index_uploaded_files(files) -> int:
    if not require_admin():
        raise PermissionError("Only administrators can upload documents.")
    if not files:
        return 0
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for uploaded in files:
        destination = UPLOAD_DIR / Path(uploaded.name).name
        destination.write_bytes(uploaded.getbuffer())
        docs.extend(load_file(destination))
    if docs:
        get_vectorstore().add_documents(docs)
    return len(docs)


def render_sources(sources):
    if not sources:
        return
    with st.expander("📚 Supporting Ministry sources"):
        seen = set()
        for source in sources:
            name, location = source.get("source", "Unknown"), source.get("location", "")
            if (name, location) in seen:
                continue
            seen.add((name, location))
            st.markdown(f"**{name}**" + (f" — {location}" if location else ""))


# ---------------------------------------------------------------------------
# Session state and sidebar
# ---------------------------------------------------------------------------
def init_state():
    defaults = {"messages": [], "conversations": [], "history_search": "",
                "thread_id": str(time.time_ns()), "active_page": "Home"}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_conversation():
    messages = list(st.session_state.get("messages", []))
    if not messages:
        return
    title = next((re.sub(r"\s+", " ", m["content"]).strip()[:48]
                  for m in messages if m.get("role") == "user"), "New conversation")
    conversations = st.session_state.setdefault("conversations", [])
    item = {"id": st.session_state.thread_id, "title": title,
            "messages": messages, "updated": time.time()}
    for i, old in enumerate(conversations):
        if old["id"] == item["id"]:
            conversations[i] = item
            break
    else:
        conversations.append(item)
    st.session_state.conversations = sorted(
        conversations, key=lambda x: x["updated"], reverse=True
    )[:30]


def new_chat():
    save_conversation()
    st.session_state.thread_id = str(time.time_ns())
    st.session_state.messages = []
    st.session_state.active_page = "Home"
    st.rerun()


def logout():
    save_conversation()
    for key in ("authenticated", "username", "role", "messages",
                "conversations", "thread_id", "history_search", "active_page"):
        st.session_state.pop(key, None)
    st.rerun()


def sidebar():
    """Render the professional MLF Chatbot sidebar and chat history."""

    with st.sidebar:
        # --------------------------------------------------------
        # MINISTRY BRANDING
        # --------------------------------------------------------
        st.markdown("""
        <div class="mlf-brand">
            <div class="mlf-brand-title">
                MLF Chatbot<br>
                Ministry of Livestock<br>
                and Fisheries
            </div>
            <div class="mlf-brand-sub">
                Better Livestock and Fisheries<br>
                for a Prosperous Tanzania
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --------------------------------------------------------
        # MAIN NAVIGATION
        # --------------------------------------------------------
        nav = [
            ("⌂", "Home"),
            ("+", "New Chat"),
            ("◷", "Chat History"),
            ("▤", "Documents"),
        ]

        for symbol, label in nav:
            # Only administrators can access Documents.
            if (
                label == "Documents"
                and st.session_state.get("role") != "admin"
            ):
                continue

            is_active_page = (
                st.session_state.get("active_page", "Home") == label
            )

            if st.button(
                f"{symbol}   {label}",
                key=f"nav_{label}",
                use_container_width=True,
                type="primary" if is_active_page else "secondary",
            ):
                if label == "New Chat":
                    new_chat()
                else:
                    st.session_state.active_page = label
                    st.rerun()

        st.divider()

        # --------------------------------------------------------
        # CHAT HISTORY
        # --------------------------------------------------------
        st.markdown("### 💬 Recent conversations")

        search_text = st.text_input(
            "Search conversations",
            key="history_search",
            placeholder="Search previous chats...",
            label_visibility="collapsed",
        )

        conversations = st.session_state.get("conversations", [])
        query = search_text.strip().lower()

        # Search conversation titles and message contents.
        filtered = [
            conversation
            for conversation in conversations
            if (
                not query
                or query in str(
                    conversation.get("title", "")
                ).lower()
                or any(
                    query in str(
                        message.get("content", "")
                    ).lower()
                    for message in conversation.get("messages", [])
                )
            )
        ]

        # Most recently updated conversations first.
        filtered.sort(
            key=lambda conversation: conversation.get("updated", 0),
            reverse=True,
        )

        # Group history by date.
        from datetime import datetime, timedelta

        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        seven_days_ago = today - timedelta(days=7)

        groups = {
            "📌 Pinned": [],
            "Today": [],
            "Yesterday": [],
            "Previous 7 Days": [],
            "Older": [],
        }

        for conversation in filtered:
            if conversation.get("pinned", False):
                groups["📌 Pinned"].append(conversation)
                continue

            try:
                updated = float(conversation.get("updated", 0))
                conversation_date = datetime.fromtimestamp(
                    updated
                ).date()
            except (TypeError, ValueError, OSError):
                conversation_date = today

            if conversation_date == today:
                group_name = "Today"
            elif conversation_date == yesterday:
                group_name = "Yesterday"
            elif conversation_date >= seven_days_ago:
                group_name = "Previous 7 Days"
            else:
                group_name = "Older"

            groups[group_name].append(conversation)

        if not filtered:
            st.caption(
                "No conversations found."
                if query
                else "Your conversations will appear here."
            )

        # Render the conversation groups.
        for group_name, group_conversations in groups.items():
            if not group_conversations:
                continue

            st.markdown(f"**{group_name}**")

            for conversation in group_conversations:
                conversation_id = conversation.get("id")

                title = str(
                    conversation.get(
                        "title", "New conversation"
                    )
                ).strip()

                is_current = (
                    conversation_id
                    == st.session_state.get("thread_id")
                )

                if len(title) > 38:
                    title = title[:35] + "..."

                if is_current:
                    button_label = f"●  {title}"
                elif conversation.get("pinned", False):
                    button_label = f"📌  {title}"
                else:
                    button_label = title

                if st.button(
                    button_label,
                    key=f"history_{conversation_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                    help="Open this conversation",
                ):
                    # Save the current chat before opening another.
                    save_conversation()

                    # Restore the selected conversation.
                    selected = next(
                        (
                            item
                            for item in st.session_state.get(
                                "conversations", []
                            )
                            if item.get("id") == conversation_id
                        ),
                        None,
                    )

                    if selected:
                        st.session_state.thread_id = selected["id"]
                        st.session_state.messages = list(
                            selected.get("messages", [])
                        )
                        st.session_state.active_page = "Home"

                    st.rerun()

                # Show a preview of the latest user question.
                messages = conversation.get("messages", [])

                last_question = next(
                    (
                        str(message.get("content", "")).strip()
                        for message in reversed(messages)
                        if message.get("role") == "user"
                    ),
                    "",
                )

                if last_question:
                    preview = " ".join(last_question.split())

                    if len(preview) > 58:
                        preview = preview[:55] + "..."

                    st.caption(preview)

        st.divider()

        # --------------------------------------------------------
        # USER PROFILE
        # --------------------------------------------------------
        username = st.session_state.get("username", "User")
        role = st.session_state.get("role", "user")

        full_name = (
            username.replace(".", " ")
            .replace("_", " ")
            .title()
        )

        role_label = (
            "Senior ICT Officer"
            if role == "admin"
            else "Standard User"
        )

        initial = (full_name[:1] or "U").upper()

        st.markdown(
            f"""
            <div class="mlf-profile">
                <div class="mlf-avatar">{initial}</div>
                <div>
                    <div class="mlf-profile-name">
                        {full_name}
                    </div>
                    <div class="mlf-profile-role">
                        {role_label}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "↪  Sign out",
            key="sidebar_logout",
            use_container_width=True,
        ):
            logout()

        st.markdown(
            """
            <div style="
                color:#a9c1d5;
                font-size:10px;
                text-align:center;
                margin-top:20px;">
                MLF Chatbot v2.0
            </div>
            """,
            unsafe_allow_html=True,
        )


def header():
    st.markdown("""
    <div class="mlf-topbar">
      <div class="mlf-bot">🤖</div>
      <div style="flex:1">
        <div class="mlf-app-title">MLF Chatbot</div>
        <div class="mlf-app-subtitle">AI Assistant for Livestock and Fisheries Services</div>
      </div>
    </div>""", unsafe_allow_html=True)


def welcome():
    st.markdown("""
    <div style="display:flex;gap:12px;align-items:flex-start;margin:8px 0 18px">
      <div class="mlf-bot">🤖</div>
      <div class="mlf-welcome">
        <p>Hello! 👋</p>
        <p>I'm your AI assistant for the Ministry of Livestock and Fisheries.</p>
        <p>I can help you with information about livestock, fisheries, permits,
        regulations, diseases, and other related services.</p>
        <p style="margin-top:14px">How can I assist you today?</p>
        <div style="font-size:10px;color:#6b7e93;margin-top:10px">Ask in English or Kiswahili</div>
      </div>
    </div>""", unsafe_allow_html=True)
    cols = st.columns(4)
    shortcuts = [
        ("🐟", "Fish Export\nPermits", "mlf-shortcut-green",
         "What are the requirements for fish export permits?"),
        ("🐄", "Livestock\nDiseases", "mlf-shortcut-blue",
         "What information is available about livestock diseases?"),
        ("▤", "Regulations\n& Policies", "mlf-shortcut-gold",
         "Summarize the fisheries regulations available in the Ministry knowledge base."),
        ("▦", "MLF\nServices", "mlf-shortcut-purple",
         "What Ministry of Livestock and Fisheries services can you help me with?"),
    ]
    for col, (icon, title, cls, query) in zip(cols, shortcuts):
        with col:
            st.markdown(f'<div class="mlf-shortcut {cls}">{icon} &nbsp; {title.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True)
            if st.button("Ask about this  ›", key=f"shortcut_{title}", use_container_width=True):
                st.session_state.pending_question = query
                st.rerun()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main():
    init_state()
    sidebar()
    header()

    page = st.session_state.get("active_page", "Home")
    if page == "Settings":
        st.subheader("⚙ Settings")
        st.info("Your account is signed in. Language selection and chat settings can be configured here in a future release.")
        return
    if page == "Help & Support":
        st.subheader("Help & Support")
        st.markdown("Ask questions about fisheries laws, permits, licensing, regulations and Ministry documents. Answers are based on documents indexed by the administrator.")
        return
    if page == "MLF Services":
        st.subheader("MLF Services")
        st.markdown("- Fish export permits\n- Fisheries regulations and policies\n- Livestock disease information\n- Ministry forms and procedures")
        st.caption("Choose a service card on Home or type a question below.")
    if page == "Documents":
        st.subheader("📚 Knowledge Centre")
        if st.session_state.get("role") != "admin":
            st.warning("Only administrators can upload and index documents.")
            return
        files = st.file_uploader(
            "Upload official Ministry documents",
            type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
            accept_multiple_files=True,
        )
        if st.button("Add to Knowledge Base", type="primary", disabled=not files):
            with st.spinner("Processing documents and creating embeddings..."):
                try:
                    count = index_uploaded_files(files)
                    st.success(f"{count} knowledge chunks indexed successfully.")
                except Exception as exc:
                    st.error("Document processing failed.")
                    st.exception(exc)
        return
    if page == "Chat History":
        st.subheader("🕘 Chat History")
        
        history = st.session_state.get("conversations", [])
        for item in history:
            if st.session_state.get("history_search", "").lower() in item.get("title", "").lower():
                if st.button(item.get("title", "Conversation"), key=f"open_{item['id']}",
                             use_container_width=True):
                    save_conversation()
                    st.session_state.thread_id = item["id"]
                    st.session_state.messages = list(item.get("messages", []))
                    st.session_state.active_page = "Home"
                    st.rerun()
        if not history:
            st.caption("Your recent conversations will appear here.")
        return

   
    if not st.session_state.messages:
        welcome()
    for message in st.session_state.messages:
        with st.chat_message(message.get("role", "assistant"),
                             avatar="🤖" if message.get("role") == "assistant" else "👤"):
            st.markdown(message.get("content", ""))
            if message.get("role") == "assistant":
                render_sources(message.get("sources", []))
   

    question = st.chat_input("Type your message here...")
    pending = st.session_state.pop("pending_question", None)
    if pending and not question:
        question = pending
    if not question or not question.strip():
        return
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🤖"):
        started = time.perf_counter()
        try:
            with st.spinner("Searching Ministry documents and preparing your answer..."):
                result = get_graph().invoke({"question": question, "attempts": 0})
            elapsed = time.perf_counter() - started
            answer = str(result.get("answer", "")).strip()
            if not answer:
                answer = "I could not generate an answer from the available Ministry knowledge base."
            st.markdown(answer)
            render_sources(result.get("sources", []))
            st.caption(f"Response time: {elapsed:.1f}s")
            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "language": result.get("language", "en"),
                "response_time": elapsed, "sources": result.get("sources", []),
            })
            save_conversation()
        except Exception as exc:
            st.error("Unable to process the request.")
            st.exception(exc)


if __name__ == "__main__":
    main()
