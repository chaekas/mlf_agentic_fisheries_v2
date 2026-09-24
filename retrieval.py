"""MLF Fisheries Regulatory Compliance Agentic AI."""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

from ingest import SUPPORTED_EXTENSIONS, collect_paths, load_file


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5433/mlf_fisheries",
)

CHECKPOINT_DATABASE_URL = os.getenv(
    "CHECKPOINT_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/mlf_fisheries",
)

COLLECTION = os.getenv(
    "PGVECTOR_COLLECTION",
    "mlf_fisheries_regulations",
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
)

TOP_K = int(
    os.getenv("TOP_K", "6")
)

MIN_RELEVANCE = float(
    os.getenv("MIN_RELEVANCE", "0.48")
)

UPLOAD_DIR = Path(
    os.getenv("UPLOAD_DIR", "data/uploads")
)


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


# ============================================================
# VECTOR STORE
# ============================================================

@st.cache_resource
def get_vectorstore():

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION,
        connection=DATABASE_URL,
        use_jsonb=True,
        create_extension=True,
    )


# ============================================================
# LLM
# ============================================================

@st.cache_resource
def get_llm():

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,
        max_retries=2,
    )


# ============================================================
# GRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):

    question: str

    language: str

    rewritten_query: str

    documents: List[Document]

    relevance: float

    answer: str

    sources: List[Dict[str, Any]]

    attempts: int

    route: str

    latency: float

    # NEW:
    # Used to identify direct form requests.
    is_form_request: bool

    # NEW:
    # Stores requested form number.
    form_number: str


# ============================================================
# FORM REQUEST DETECTION
# ============================================================

def detect_form_request(question: str):

    """
    Detect whether the user is asking for a specific form.

    Examples:

        Fomu Na. 1
        Fomu Na 1
        Fomu namba 1
        Fomu 1
    """

    question_lower = question.lower()

    patterns = [
        r"fomu\s*na\.?\s*(\d+)",
        r"fomu\s*namba\s*(\d+)",
        r"fomu\s*(\d+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question_lower
        )

        if match:

            return True, match.group(1)

    return False, ""


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(state: AgentState):

    question = state["question"].lower()

    sw = [
        "naomba",
        "nahitaji",
        "fomu",
        "vibali",
        "uvuvi",
        "samaki",
        "mvuvi",
        "nyavu",
        "leseni",
        "marufuku",
        "msimu",
        "adhabu",
        "usafirishaji",
        "mamlaka",
        "hifadhi",
    ]

    score = sum(
        word in question
        for word in sw
    )

    language = "sw" if score else "en"

    # --------------------------------------------------------
    # Detect direct form request
    # --------------------------------------------------------

    is_form_request, form_number = detect_form_request(
        state["question"]
    )

    return {
        "language": language,
        "is_form_request": is_form_request,
        "form_number": form_number,
    }


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(state: AgentState):

    start = time.perf_counter()

    question = state["question"]

    is_form_request = state.get(
        "is_form_request",
        False
    )

    form_number = state.get(
        "form_number",
        ""
    )

    # ========================================================
    # NORMAL RETRIEVAL
    # ========================================================

    if not is_form_request:

        query = state.get(
            "rewritten_query"
        ) or question

        try:

            results = (
                get_vectorstore()
                .similarity_search_with_relevance_scores(
                    query,
                    k=TOP_K
                )
            )

        except Exception as exc:

            raise RuntimeError(
                "Vector retrieval failed. "
                "Confirm that documents have been indexed "
                f"with ingest.py. Database error: {exc}"
            ) from exc

        documents = [
            doc
            for doc, score in results
            if float(score) >= MIN_RELEVANCE
        ]

        relevance = max(
            (
                float(score)
                for _, score in results
            ),
            default=0.0
        )

        return {
            "documents": documents,
            "relevance": relevance,
            "latency": (
                time.perf_counter()
                - start
            ),
        }

    # ========================================================
    # DIRECT FORM RETRIEVAL
    # ========================================================

    # We deliberately retrieve more candidates.
    #
    # The previous test showed that Fomu Na. 1 was already
    # indexed but ranked around result #12.
    #
    # Therefore TOP_K=6 was not enough.
    # ========================================================

    form_query = (
        f"FOMU Na. {form_number} "
        f"KIBALI CHA SHUGHULI ZA UVUVI "
        f"KATIKA MAENEO YA HIFADHI "
        f"JEDWALI LA KWANZA"
    )

    try:

        results = (
            get_vectorstore()
            .similarity_search_with_relevance_scores(
                form_query,
                k=30
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Form retrieval failed. "
            f"Database error: {exc}"
        ) from exc

    # ========================================================
    # EXACT FORM MATCHING
    # ========================================================

    exact_matches = []

    related_matches = []

    for doc, score in results:

        content = (
            doc.page_content
            or ""
        )

        content_lower = content.lower()

        # ----------------------------------------------------
        # Exact form number
        # ----------------------------------------------------

        form_patterns = [
            f"fomu na. {form_number}",
            f"fomu na {form_number}",
            f"fomu namba {form_number}",
        ]

        exact_number = any(
            pattern in content_lower
            for pattern in form_patterns
        )

        # ----------------------------------------------------
        # Important form terminology
        # ----------------------------------------------------

        has_jedwali = (
            "jedwali la kwanza"
            in content_lower
        )

        has_form_title = (
            "kibali cha shughuli za uvuvi"
            in content_lower
        )

        # ----------------------------------------------------
        # Strong match
        # ----------------------------------------------------

        if (
            exact_number
            and (
                has_form_title
                or has_jedwali
            )
        ):

            exact_matches.append(
                (
                    doc,
                    float(score)
                )
            )

        elif exact_number:

            related_matches.append(
                (
                    doc,
                    float(score)
                )
            )

    # ========================================================
    # FALLBACK
    # ========================================================

    if exact_matches:

        selected = exact_matches

    elif related_matches:

        selected = related_matches

    else:

        selected = results[:TOP_K]

    # ========================================================
    # KEEP A SMALL NUMBER OF DOCUMENTS
    # ========================================================

    documents = [
        doc
        for doc, score in selected[:4]
    ]

    relevance = max(
        (
            score
            for _, score in selected
        ),
        default=0.0
    )

    return {
        "documents": documents,
        "relevance": relevance,
        "latency": (
            time.perf_counter()
            - start
        ),
    }


# ============================================================
# DOCUMENT GRADING
# ============================================================

def grade_documents(state: AgentState):

    # --------------------------------------------------------
    # IMPORTANT:
    # Never rewrite an exact form request.
    # --------------------------------------------------------

    if state.get(
        "is_form_request",
        False
    ):

        return {
            "route": "generate"
        }

    # --------------------------------------------------------
    # Normal questions
    # --------------------------------------------------------

    if not state.get("documents"):

        if state.get(
            "attempts",
            0
        ) < 1:

            return {
                "route": "rewrite"
            }

        return {
            "route": "generate"
        }

    if (
        state.get(
            "relevance",
            0
        ) < MIN_RELEVANCE
        and
        state.get(
            "attempts",
            0
        ) < 1
    ):

        return {
            "route": "rewrite"
        }

    return {
        "route": "generate"
    }


# ============================================================
# QUERY REWRITE
# ============================================================

def rewrite_query(state: AgentState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
Rewrite the user's fisheries-regulation question
into a precise retrieval query.

Preserve Tanzanian legal, regulatory, permit,
fisheries and enforcement terms.

Return only the rewritten query.
""",
        ),
        (
            "human",
            "{question}"
        ),
    ])

    response = get_llm().invoke(
        prompt.format_messages(
            question=state["question"]
        )
    )

    return {
        "rewritten_query": response.content.strip(),
        "attempts": (
            state.get(
                "attempts",
                0
            ) + 1
        ),
    }


# ============================================================
# GENERATE FINAL ANSWER
# ============================================================

def generate(state: AgentState):

    documents = state.get(
        "documents",
        []
    )

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []

    for i, doc in enumerate(
        documents,
        1
    ):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            doc.metadata.get(
                "slide",
                "?"
            )
        )

        context_parts.append(
            f"[{i}] {doc.page_content}\n"
            f"Source: {source} "
            f"page {page}"
        )

    context = "\n\n".join(
        context_parts
    )

    # ========================================================
    # NO CONTEXT
    # ========================================================

    if not context:

        return {
            "answer": (
                "I could not find sufficient "
                "information in the Ministry "
                "knowledge base to answer this question."
            ),
            "sources": [],
        }

    # ========================================================
    # LANGUAGE
    # ========================================================

    language = (
        "Swahili"
        if state.get("language") == "sw"
        else "English"
    )

    # ========================================================
    # FORM REQUEST
    # ========================================================

    if state.get(
        "is_form_request",
        False
    ):

        form_number = state.get(
            "form_number",
            ""
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are the MLF Fisheries Regulatory
Compliance AI Assistant.

The user is requesting a specific official
form from the Ministry knowledge base.

IMPORTANT RULES:

1. Answer ONLY from the supplied context.
2. Do NOT substitute another form.
3. Do NOT confuse Fomu Na. 1 with Fomu Na. 2,
   Fomu Na. 3, Fomu Na. 4 or Fomu Na. 5.
4. The requested form number is {form_number}.
5. If the exact requested form is present,
   identify it clearly.
6. If the actual form fields are present in
   the context, reproduce the form content
   faithfully.
7. Do not invent missing fields.
8. Do not explain another form as a replacement.
9. Answer in {language}.
10. Be direct and concise.

If the exact requested form is NOT present
in the supplied context, say that the exact
form could not be retrieved from the supplied
knowledge-base context.
""",
            ),
            (
                "human",
                """
User request:

{question}

Requested form number:

Fomu Na. {form_number}

Knowledge-base context:

{context}
""",
            ),
        ])

        response = get_llm().invoke(
            prompt.format_messages(
                language=language,
                question=state["question"],
                form_number=form_number,
                context=context,
            )
        )

    # ========================================================
    # NORMAL QUESTION
    # ========================================================

    else:

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are the MLF Fisheries Regulatory
Compliance AI Assistant.

Answer ONLY from the supplied Ministry
knowledge-base context.

Do not invent laws, penalties, fees,
dates, permits, closed seasons,
prohibited gear, or procedures.

If the context is insufficient,
clearly say so.

Answer in {language}.

Be concise but useful.

Where possible:

1. Give the direct answer.
2. Explain the applicable requirement.
3. Mention conditions or exceptions only
   when present in the context.
4. Cite supporting context as [1], [2], etc.

Do not present user claims as official facts.
""",
            ),
            (
                "human",
                """
Question:

{question}

Knowledge base:

{context}
""",
            ),
        ])

        response = get_llm().invoke(
            prompt.format_messages(
                language=language,
                question=state["question"],
                context=context,
            )
        )

    # ========================================================
    # SOURCES
    # ========================================================

    sources = [
        {
            "source": doc.metadata.get(
                "source",
                "Unknown"
            ),
            "page": doc.metadata.get(
                "page",
                doc.metadata.get(
                    "slide",
                    "?"
                )
            ),
            "lang": doc.metadata.get(
                "lang",
                ""
            ),
        }
        for doc in documents
    ]

    return {
        "answer": response.content.strip(),
        "sources": sources,
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

@st.cache_resource
def get_graph():

    saver_context = (
        PostgresSaver.from_conn_string(
            CHECKPOINT_DATABASE_URL
        )
    )

    saver = (
        saver_context.__enter__()
    )

    saver.setup()

    builder = StateGraph(
        AgentState
    )

    builder.add_node(
        "language",
        detect_language
    )

    builder.add_node(
        "retrieve",
        retrieve
    )

    builder.add_node(
        "grade",
        grade_documents
    )

    builder.add_node(
        "rewrite",
        rewrite_query
    )

    builder.add_node(
        "generate",
        generate
    )

    builder.set_entry_point(
        "language"
    )

    builder.add_edge(
        "language",
        "retrieve"
    )

    builder.add_edge(
        "retrieve",
        "grade"
    )

    builder.add_conditional_edges(
        "grade",
        lambda s: s["route"],
        {
            "rewrite": "rewrite",
            "generate": "generate",
        },
    )

    builder.add_edge(
        "rewrite",
        "retrieve"
    )

    builder.add_edge(
        "generate",
        END
    )

    return builder.compile(
        checkpointer=saver
    )


# ============================================================
# RENDER SOURCES
# ============================================================

def render_sources(sources):

    if not sources:
        return

    with st.expander(
        "📚 Sources"
    ):

        for index, source in enumerate(
            sources,
            1
        ):

            st.write(
                f"[{index}] "
                f"{source['source']} "
                f"— page/slide "
                f"{source['page']}"
            )


# ============================================================
# INDEX UPLOADED FILES
# ============================================================

def index_uploaded_files(
    uploaded_files
):

    if not uploaded_files:
        return 0

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    paths = []

    for uploaded in uploaded_files:

        destination = (
            UPLOAD_DIR
            / uploaded.name
        )

        destination.write_bytes(
            uploaded.getbuffer()
        )

        paths.append(
            destination
        )

    documents: List[Document] = []

    for path in paths:

        documents.extend(
            load_file(path)
        )

    if not documents:
        return 0

    get_vectorstore().add_documents(
        documents
    )

    return len(documents)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.set_page_config(
        page_title="MLF Fisheries Agentic AI",
        page_icon="🐟",
        layout="wide",
    )

    st.title(
        "🐟 MLF Fisheries Regulatory Compliance — Agentic AI"
    )

    st.caption(
        "LangGraph + PostgreSQL/pgvector + Groq + BGE-M3"
    )

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    if "thread_id" not in st.session_state:

        st.session_state.thread_id = (
            f"user-{int(time.time())}"
        )

    if "messages" not in st.session_state:

        st.session_state.messages = []

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.subheader(
            "⚙️ Configuration"
        )

        st.write(
            f"Model: `{GROQ_MODEL}`"
        )

        st.write(
            f"Embedding: `{EMBEDDING_MODEL}`"
        )

        st.write(
            f"Top-K: `{TOP_K}`"
        )

        st.write(
            "Database: `PostgreSQL + pgvector`"
        )

        if st.button(
            "🧹 New conversation"
        ):

            st.session_state.thread_id = (
                f"user-{int(time.time())}"
            )

            st.session_state.messages = []

            st.rerun()

        st.divider()

        st.subheader(
            "📂 Knowledge Base"
        )

        st.caption(
            "Upload supported documents and images. "
            "Text is extracted and indexed into pgvector."
        )

        uploaded_files = st.file_uploader(
            "Upload files",
            type=[
                ext.lstrip(".")
                for ext in sorted(
                    SUPPORTED_EXTENSIONS
                )
            ],
            accept_multiple_files=True,
        )

        if st.button(
            "📥 Index uploaded files",
            disabled=not uploaded_files
        ):

            with st.spinner(
                "Extracting text, creating embeddings and indexing…"
            ):

                try:

                    count = index_uploaded_files(
                        uploaded_files
                    )

                    st.success(
                        f"Indexed {count} text chunks into the knowledge base."
                    )

                except Exception as exc:

                    st.error(
                        f"Indexing failed: {exc}"
                    )

    # --------------------------------------------------------
    # PREVIOUS MESSAGES
    # --------------------------------------------------------

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if message.get(
                "sources"
            ):

                render_sources(
                    message["sources"]
                )

    # --------------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask in English or Swahili…"
    )

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # PROCESS QUESTION
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Retrieving Ministry information..."
        ):

            start = time.perf_counter()

            try:

                result = get_graph().invoke(
                    {
                        "question": question,
                        "attempts": 0,
                    },
                    config={
                        "configurable": {
                            "thread_id":
                                st.session_state.thread_id
                        }
                    },
                )

            except Exception as exc:

                st.error(
                    str(exc)
                )

                return

            elapsed = (
                time.perf_counter()
                - start
            )

        st.markdown(
            result["answer"]
        )

        st.caption(
            f"⚡ {elapsed:.2f}s | "
            f"Retrieval relevance: "
            f"{result.get('relevance', 0):.2f} | "
            f"Language: "
            f"{result.get('language', 'en').upper()}"
        )

        render_sources(
            result.get(
                "sources",
                []
            )
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get(
                    "sources",
                    []
                ),
            }
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()