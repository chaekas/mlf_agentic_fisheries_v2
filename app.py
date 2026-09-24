"""
MLF Fisheries Regulatory Compliance Agentic AI

Architecture:
    Streamlit
        ↓
    LangGraph
        ↓
    Language Detection
        ↓
    PostgreSQL + PGVector Retrieval
        ↓
    Relevance Grading
        ↓
    Query Rewrite (if required)
        ↓
    Ollama LLM
        ↓
    Answer + Sources

Database:
    Docker PostgreSQL + pgvector
    Host: localhost
    Port: 5433

Existing Windows PostgreSQL on port 5432 is not affected.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import re
import time

from pathlib import Path
from typing import Any, Dict, List, TypedDict

import streamlit as st

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate



from langchain_huggingface import HuggingFaceEmbeddings

from langchain_postgres import PGVector

from langgraph.graph import END, StateGraph

from langgraph.checkpoint.postgres import PostgresSaver
from langchain_ollama import ChatOllama


from ingest import (
    SUPPORTED_EXTENSIONS,
    load_file,
)


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 3. DATABASE CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")

CHECKPOINT_DATABASE_URL = os.getenv(
    "CHECKPOINT_DATABASE_URL"
)

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing from the .env file."
    )

if not CHECKPOINT_DATABASE_URL:
    raise RuntimeError(
        "CHECKPOINT_DATABASE_URL is missing from the .env file."
    )


# ============================================================
# 4. APPLICATION CONFIGURATION
# ============================================================

COLLECTION = os.getenv(
    "PGVECTOR_COLLECTION",
    "mlf_fisheries_regulations",
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
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
# 5. EMBEDDING MODEL
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
# 6. PGVECTOR DATABASE
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
# 7. OLLAMA LLM
# ============================================================

@st.cache_resource
def get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=250,
        keep_alive="10m",
    )

# ============================================================
# 8. LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):

    question: str

    language: str

    intent: str

    rewritten_query: str

    documents: List[Dict[str, Any]]

    relevance: float

    answer: str

    sources: List[Dict[str, Any]]

    attempts: int

    route: str

    latency: float


# ============================================================
# 9. LANGUAGE AND INTENT DETECTION
# ============================================================

def detect_language(
    state: AgentState
):
    """
    Detect the user's language and whether the message
    is casual conversation or a fisheries-regulatory query.

    Casual messages are handled directly and are NOT sent
    to PGVector. This prevents greetings, thanks and
    farewell messages such as "kwaheri" from retrieving
    unrelated regulatory documents.
    """

    question = state["question"].strip().lower()

    # --------------------------------------------------------
    # Swahili vocabulary
    # --------------------------------------------------------

    swahili_words = [
        "naomba",
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
        "kanuni",
        "sheria",
        "kibali",
        "mazao",
        "uhifadhi",
        "kwaheri",
        "asante",
        "habari",
        "maana",
        "unaelewa",
        "hapana",
        "ndiyo",
        "karibu",
        "inatosha",
        "tutaonana",
        "tafadhali",
        "nini",
        "vipi",
        "lini",
        "wapi",
        "kwa nini",
        "je",
        "masharti",
        "chombo",
        "eneo",
        "hifadhi",
        "kutolewa",
        "kutoa",
    ]

    # Use word/phrase matching rather than substring matching
    score = 0

    for word in swahili_words:
        if " " in word:
            if word in question:
                score += 1
        else:
            if re.search(
                rf"\b{re.escape(word)}\b",
                question,
            ):
                score += 1

    language = "sw" if score > 0 else "en"

    # --------------------------------------------------------
    # Detect casual conversation
    # --------------------------------------------------------

    regulatory_terms = [
        "kibali",
        "vibali",
        "uvuvi",
        "samaki",
        "mvuvi",
        "kanuni",
        "sheria",
        "leseni",
        "permit",
        "licence",
        "license",
        "fishing",
        "regulation",
        "regulations",
        "form",
        "fomu",
        "masharti",
        "utaratibu",
        "ada",
        "adhabu",
        "usafirishaji",
        "chombo cha uvuvi",
        "eneo la hifadhi",
        "protected area",
    ]

    is_regulatory = any(
        term in question
        for term in regulatory_terms
    )

    casual_exact = {
        "kwaheri",
        "bye",
        "goodbye",
        "see you",
        "tutaonana",
        "asante",
        "thanks",
        "thank you",
        "habari",
        "hello",
        "hi",
        "mambo",
        "karibu",
        "shikamoo",
    }

    farewell_patterns = [
        "asante kwaheri",
        "hapana asante kwaheri",
        "asante, kwaheri",
        "thanks goodbye",
        "thanks, goodbye",
    ]

    is_casual = (
        not is_regulatory
        and (
            question in casual_exact
            or any(
                pattern in question
                for pattern in farewell_patterns
            )
            or any(
                phrase in question
                for phrase in [
                    "asante",
                    "thanks",
                    "thank you",
                    "kwaheri",
                    "goodbye",
                    "bye",
                    "tutaonana",
                ]
            )
            or (
                "kwaheri" in question
                and (
                    "maana" in question
                    or "unaelewa" in question
                    or "nini" in question
                )
            )
        )
    )

    intent = "casual" if is_casual else "regulatory"

    print(
        f"🧭 Language: {language.upper()} | "
        f"Intent: {intent}"
    )

    return {
        "language": language,
        "intent": intent,
    }


# ============================================================
# 10. ROUTE CASUAL OR REGULATORY QUESTIONS
# ============================================================

def route_after_language(
    state: AgentState
):
    """
    Route casual conversation directly to a response node.
    Regulatory questions continue to retrieval.
    """

    if state.get("intent") == "casual":
        return "direct"

    return "retrieve"


# ============================================================
# 11. DIRECT CASUAL RESPONSE
# ============================================================

def direct_response(
    state: AgentState
):
    """
    Respond to simple conversational messages without
    querying the fisheries knowledge base.
    """

    question = state["question"].strip().lower()
    language = state.get("language", "sw")

    # --------------------------------------------------------
    # Meaning of "kwaheri"
    # --------------------------------------------------------

    if "kwaheri" in question and (
        "maana" in question
        or "unaelewa" in question
        or "nini" in question
    ):
        if language == "sw":
            answer = (
                'Ndiyo. "Kwaheri" ni neno la kumuaga mtu, '
                'likimaanisha "goodbye" au "farewell".'
            )
        else:
            answer = (
                '"Kwaheri" is a Swahili word used when '
                'saying goodbye or farewell to someone.'
            )

        return {
            "answer": answer,
            "sources": [],
            "relevance": 0.0,
        }

    # --------------------------------------------------------
    # Farewell
    # --------------------------------------------------------

    if (
        "kwaheri" in question
        or question in {
            "bye",
            "goodbye",
            "see you",
            "tutaonana",
        }
    ):
        if language == "sw":
            answer = "Karibu! Kwaheri."
        else:
            answer = "You're welcome. Goodbye!"

        return {
            "answer": answer,
            "sources": [],
            "relevance": 0.0,
        }

    # --------------------------------------------------------
    # Thanks
    # --------------------------------------------------------

    if (
        "asante" in question
        or "thanks" in question
        or "thank you" in question
    ):
        if language == "sw":
            answer = "Karibu!"
        else:
            answer = "You're welcome!"

        return {
            "answer": answer,
            "sources": [],
            "relevance": 0.0,
        }

    # --------------------------------------------------------
    # Greetings
    # --------------------------------------------------------

    if question in {
        "habari",
        "hello",
        "hi",
        "mambo",
        "shikamoo",
    }:
        if language == "sw":
            answer = "Habari! Karibu. Naweza kukusaidia kuhusu masuala ya uvuvi na kanuni zake."
        else:
            answer = "Hello! How can I help you with fisheries regulations?"

        return {
            "answer": answer,
            "sources": [],
            "relevance": 0.0,
        }

    # Fallback for a casual message
    if language == "sw":
        answer = "Karibu! Naweza kukusaidia."
    else:
        answer = "You're welcome! I can help you."

    return {
        "answer": answer,
        "sources": [],
        "relevance": 0.0,
    }


# ============================================================
# 12. FORM NUMBER DETECTION
# ============================================================

# 13. FORM NUMBER DETECTION
# ============================================================

def detect_requested_form(
    query: str
):
    """
    Detect a specific form only when the
    user explicitly asks for it.

    Examples:
        Fomu Na. 1
        Fomu Na 1
        Fomu namba 1
        Fomu 1

    Returns:
        "1"

    or:

        None
    """

    pattern = (
        r"\bfomu\s*(?:na\.?|namba)?\s*(\d+)\b"
    )

    match = re.search(
        pattern,
        query.lower()
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# 14. DOCUMENT RETRIEVAL
# ============================================================

def retrieve(
    state: AgentState
) -> AgentState:

    """
    Retrieve relevant documents from PostgreSQL/PGVector.

    Retrieval strategy:

    1. Semantic vector similarity
    2. Query-term matching
    3. Explicit Form Number matching
    4. Heading/title relevance
    5. Penalize obvious form mismatch
    6. Return only the strongest documents

    IMPORTANT:

    Form-specific ranking is activated ONLY when the
    user explicitly asks for a form.

    Normal questions remain semantic retrieval questions.
    """

    question = state["question"]

    query = (
        state.get("rewritten_query")
        or question
    )
    retrieve_start = time.perf_counter()
    vectorstore = get_vectorstore()

    # --------------------------------------------------------
    # 11.1 Retrieve a larger candidate set
    # --------------------------------------------------------

    candidate_k = max(
        TOP_K * 5,
        30
    )

    try:

        results = (
            vectorstore
            .similarity_search_with_relevance_scores(
                query,
                k=candidate_k,
            )
        )
        retrieve_elapsed = (
           time.perf_counter() - retrieve_start
        )

        print(
          f"⏱️ PGVector retrieval: "
          f"{retrieve_elapsed:.2f}s"
        )

    except Exception as exc:

        print(
            f"Retrieval error: {exc}"
        )

        return {
            **state,
            "documents": [],
            "relevance": 0.0,
        }

    # --------------------------------------------------------
    # 11.2 No results
    # --------------------------------------------------------

    if not results:

        return {
            **state,
            "documents": [],
            "relevance": 0.0,
        }

    # --------------------------------------------------------
    # 11.3 Query terms
    # --------------------------------------------------------

    query_lower = query.lower()

    stop_words = {

        "na",
        "ya",
        "wa",
        "ni",
        "je",
        "kwa",
        "katika",
        "cha",
        "za",
        "au",
        "hii",
        "hiyo",
        "hizi",
        "hizo",

        "the",
        "is",
        "are",
        "of",
        "in",
        "to",
        "for",
        "what",
        "which",
        "how",
        "who",
        "a",
        "an",
    }

    query_terms = set()

    for word in query_lower.split():

        clean_word = word.strip(
            ".,:;!?()[]{}\"'"
        )

        if (
            len(clean_word) >= 3
            and clean_word not in stop_words
        ):

            query_terms.add(
                clean_word
            )

    # --------------------------------------------------------
    # 11.4 Detect explicit form request
    # --------------------------------------------------------

    requested_form = detect_requested_form(
        query
    )

    # --------------------------------------------------------
    # 11.5 Ranking
    # --------------------------------------------------------

    ranked = []

    for original_rank, (
        doc,
        vector_score
    ) in enumerate(results):

        content = (
            doc.page_content
            or ""
        )

        content_lower = (
            content.lower()
        )

        # ----------------------------------------------------
        # Base semantic score
        # ----------------------------------------------------

        score = float(
            vector_score
        )

        # ----------------------------------------------------
        # Query-term matching
        # ----------------------------------------------------

        if query_terms:

            matched_terms = sum(

                1

                for term in query_terms

                if term in content_lower
            )

            term_ratio = (
                matched_terms
                / len(query_terms)
            )

            score += (
                term_ratio * 0.20
            )

        # ----------------------------------------------------
        # Explicit form matching
        # ----------------------------------------------------

        if requested_form:

            exact_form_pattern = (

                  r"\bfomu\s*(?:na\.?|namba)?\s*"
              rf"{re.escape(requested_form)}\b"
            )

            if re.search(
                exact_form_pattern,
                content_lower,
            ):

                score += 0.35

                # Stronger reward when form is a heading

                heading_patterns = [

                    f"fomu na. {requested_form}",

                    f"fomu na {requested_form}",

                    f"fomu namba {requested_form}",

                    f"fomu {requested_form}",
                ]

                if any(
                    pattern in content_lower
                    for pattern
                    in heading_patterns
                ):

                    score += 0.20

        # ----------------------------------------------------
        # Heading/title relevance
        # ----------------------------------------------------

        heading_terms = [

            "jedwali la kwanza",

            "fomu na.",

            "fomu namba",

            "kibali",

            "maombi",

            "rufaa",

            "kanuni",

            "sheria",

            "masharti",

            "utaratibu",
        ]

        heading_matches = sum(

            1

            for term in heading_terms

            if term in content_lower
        )

        score += min(
            heading_matches * 0.03,
            0.15,
        )

        # -----------------------------------------------------
        # Penalize obvious mismatch for an explicit form request
        # -----------------------------------------------------
        if requested_form:

            other_forms = []

            form_pattern = r"\bfomu\s*(?:na\.?|namba)?\s*(\d+)\b"

            match_list = re.finditer(form_pattern, content_lower)

            for match in match_list:
                other_forms.append(match.group(1))

            other_forms = set(other_forms)

            # If this chunk talks about another form but does not
            # contain the requested form, reduce its priority.
            if (
                other_forms
                and requested_form not in other_forms
            ):
                score -= 0.25

        # ----------------------------------------------------
        # Store ranked document
        # ----------------------------------------------------

        ranked.append({

            "doc": doc,

            "score": score,

            "vector_score":
                float(vector_score),

            "original_rank":
                original_rank,
        })

    # ========================================================
    # 11.6 Sort
    # ========================================================

    ranked.sort(
        key=lambda item:
            item["score"],
        reverse=True,
    )

    # ========================================================
    # 11.7 Select top documents
    # ========================================================

    selected = ranked[:TOP_K]

    documents = []

    for item in selected:

        doc = item["doc"]

        documents.append({

            "content":
                doc.page_content,

            "metadata":
                doc.metadata,

            "score":
                item["score"],

            "vector_score":
                item["vector_score"],
        })

    # ========================================================
    # 11.8 Best relevance
    # ========================================================

    best_score = (

        selected[0]["score"]

        if selected

        else 0.0
    )

    # ========================================================
    # 11.9 Debug logging
    # ========================================================

    print(
        "\n================ RETRIEVAL ================"
    )

    print(
        f"Query: {query}"
    )

    print(
        f"Candidates retrieved: "
        f"{len(results)}"
    )

    print(
        f"Documents selected: "
        f"{len(documents)}"
    )

    print(
        f"Requested form: "
        f"{requested_form}"
    )

    print(
        f"Best score: "
        f"{best_score:.4f}"
    )

    for i, item in enumerate(
        selected,
        start=1,
    ):

        content_preview = (

            item["doc"]
            .page_content
            .replace("\n", " ")
            .strip()
        )

        print(

            f"\n[{i}] "
            f"final={item['score']:.4f} "
            f"vector="
            f"{item['vector_score']:.4f}"
        )

        print(
            content_preview[:300]
        )

    print(
        "============================================\n"
    )

    return {

        **state,

        "documents":
            documents,

        "relevance":
            best_score,
    }


# ============================================================
# 15. GRADE RETRIEVED DOCUMENTS
# ============================================================

def grade_documents(
    state: AgentState
):

    documents = state.get(
        "documents",
        []
    )

    attempts = state.get(
        "attempts",
        0
    )

    relevance = state.get(
        "relevance",
        0.0
    )

    # --------------------------------------------------------
    # No documents
    # --------------------------------------------------------

    if not documents:

        if attempts < 1:

            return {
                "route": "rewrite"
            }

        return {
            "route": "generate"
        }

    # --------------------------------------------------------
    # Low relevance
    # --------------------------------------------------------

    if (

        relevance < MIN_RELEVANCE

        and attempts < 1

    ):

        return {
            "route": "rewrite"
        }

    # --------------------------------------------------------
    # Good retrieval
    # --------------------------------------------------------

    return {
        "route": "generate"
    }


# ============================================================
# 16. QUERY REWRITE
# ============================================================

def rewrite_query(
    state: AgentState
):

    prompt = ChatPromptTemplate.from_messages([

        (

            "system",

            """
Rewrite the user's fisheries-regulation question
into a precise retrieval query.

Preserve:

- Tanzanian fisheries legal terminology
- Regulations
- Fisheries permits
- Licences
- Enforcement
- Fishing activities
- Protected areas
- Forms
- Procedures
- Conditions
- Penalties

If the user explicitly mentions a form number,
preserve the exact form number.

Return ONLY the rewritten retrieval query.
""",
        ),

        (
            "human",
            "{question}"
        ),
    ])
    rewrite_start = time.perf_counter()
    response = get_llm().invoke(

        prompt.format_messages(

            question=
                state["question"]
        )
    )

    rewrite_elapsed = (
     time.perf_counter() - rewrite_start
    )

    print(
      f"⏱️ Ollama query rewrite: "
      f"{rewrite_elapsed:.2f}s"
    )

    return {

        "rewritten_query":
            response.content.strip(),

        "attempts":
            state.get(
                "attempts",
                0
            ) + 1,
    }


# ============================================================
# ============================================================
# 17. GENERATE FINAL ANSWER
# ============================================================

def generate(
    state: AgentState
):
    """
    Generate a regulatory answer using only retrieved evidence.
    """

    documents = state.get(
        "documents",
        []
    )

    # --------------------------------------------------------
    # Build compact and unique context
    # --------------------------------------------------------

    context_parts = []
    seen_content = set()

    for doc in documents:

        metadata = doc.get(
            "metadata",
            {}
        )

        content = doc.get(
            "content",
            ""
        ).strip()

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            metadata.get(
                "slide",
                "?"
            )
        )

        if not content:
            continue

        normalized_content = " ".join(
            content.split()
        )

        if normalized_content in seen_content:
            continue

        seen_content.add(
            normalized_content
        )

        context_parts.append(
            f"[{len(context_parts) + 1}] {content}\n"
            f"Source: {source} "
            f"page {page}"
        )

    context = "\n\n".join(
        context_parts
    )

    # Keep the final prompt compact.
    MAX_CONTEXT_CHARS = 4000

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    print(
        "\n================ FINAL CONTEXT ================"
    )

    print(
        "Original documents:",
        len(documents)
    )

    print(
        "Unique documents:",
        len(context_parts)
    )

    print(
        "Context characters:",
        len(context)
    )

    print(
        "Context words:",
        len(context.split())
    )

    print(
        "================================================"
    )

    # --------------------------------------------------------
    # No useful context
    # --------------------------------------------------------

    if not context:

        return {
            "answer": (
                "Taarifa hiyo haikupatikana "
                "katika nyaraka zilizopo kwenye "
                "kanzidata ya maarifa ya Wizara."
            ),
            "sources": [],
            "relevance": state.get(
                "relevance",
                0.0
            ),
        }

    # --------------------------------------------------------
    # Response language
    # --------------------------------------------------------

    language = (
        "Swahili"
        if state.get("language") == "sw"
        else "English"
    )

    # ========================================================
    # FINAL ANSWER PROMPT
    # ========================================================

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are the MLF Fisheries Regulatory Compliance AI Assistant.

You answer regulatory questions ONLY from the provided
Ministry knowledge-base context.

STRICT EVIDENCE RULES:

1. Never invent information.

2. Never add facts that are not explicitly supported
   by the supplied context.

3. Never invent a law, regulation, rule, permit, fee,
   penalty, date, procedure or condition.

4. Do not interpret a legal provision beyond what is
   explicitly supported by the supplied context.

5. If the context does not contain enough information,
   clearly say that the information was not found in
   the available Ministry knowledge base.

IMPORTANT RULE FOR NUMBERED PROVISIONS:

Legal documents often contain numbers such as:
10., 12., (1), (a), (b), (c).

These are legal provision, section, sub-rule or item
numbers. Do NOT treat them as quantities.

IMPORTANT QUESTION-MATCHING RULE:

Answer the exact question asked.

If the user asks for conditions, prioritize the actual
conditions supported by the relevant provision.

Do not unnecessarily mix conditions with application
procedures, fees, validity, possession requirements,
other forms or unrelated provisions.

ANSWER STRUCTURE:

- Give a short direct answer.
- List conditions separately when appropriate.
- Preserve the meaning of the official document.
- Do not create additional conditions.
- Distinguish different legal provisions clearly.
- Cite supporting context using [1], [2], [3], etc.

Answer in {language}.

Be concise, precise and suitable for a Ministry
regulatory compliance assistant.
""",
        ),
        (
            "human",
            """
Question:
{question}

Knowledge Base:
{context}

Provide the answer based ONLY on the knowledge base above.
""",
        ),
    ])

    # --------------------------------------------------------
    # Prepare prompt once
    # --------------------------------------------------------

    final_messages = prompt.format_messages(
        language=language,
        question=state["question"],
        context=context,
    )

    prompt_text = "\n".join(
        message.content
        for message in final_messages
    )

    print(
        "\n================ FINAL PROMPT ================"
    )

    print(
        "Prompt characters:",
        len(prompt_text)
    )

    print(
        "Prompt words:",
        len(prompt_text.split())
    )

    print(
        "Context characters:",
        len(context)
    )

    print(
        "Context words:",
        len(context.split())
    )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # Call Ollama ONCE
    # --------------------------------------------------------

    llm_start = time.perf_counter()

    response = get_llm().invoke(
        final_messages
    )

    llm_elapsed = (
        time.perf_counter()
        - llm_start
    )

    print(
        f"⏱️ Ollama final answer: "
        f"{llm_elapsed:.2f}s"
    )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = []
    seen_sources = set()

    for doc in documents:

        metadata = doc.get(
            "metadata",
            {}
        )

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            metadata.get(
                "slide",
                "?"
            )
        )

        source_key = (
            source,
            page
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(
            source_key
        )

        sources.append({
            "source": source,
            "page": page,
            "lang": metadata.get(
                "lang",
                ""
            ),
        })

    return {
        "answer": response.content.strip(),
        "sources": sources,
        "relevance": state.get(
            "relevance",
            0.0
        ),
    }


# ============================================================
# 18. BUILD LANGGRAPH
# ============================================================

@st.cache_resource
def get_graph():

    # --------------------------------------------------------
    # PostgreSQL connection pool
    # --------------------------------------------------------

    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=CHECKPOINT_DATABASE_URL,
        min_size=1,
        max_size=5,
        open=True,
    )

    # --------------------------------------------------------
    # LangGraph PostgreSQL checkpointer
    # --------------------------------------------------------

    saver = PostgresSaver(pool)

    saver.setup()

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    builder = StateGraph(AgentState)

    builder.add_node(
        "language",
        detect_language
    )

    builder.add_node(
        "direct_response",
        direct_response
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

    # --------------------------------------------------------
    # Graph flow
    # --------------------------------------------------------

    builder.set_entry_point(
        "language"
    )

    # Language/intent detection decides whether retrieval
    # is necessary.
    builder.add_conditional_edges(
        "language",
        route_after_language,
        {
            "direct": "direct_response",
            "retrieve": "retrieve",
        },
    )

    builder.add_edge(
        "direct_response",
        END
    )

    builder.add_edge(
        "retrieve",
        "grade"
    )

    builder.add_conditional_edges(
        "grade",
        lambda state: state["route"],
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


# 19. RENDER SOURCES
# ============================================================

def render_sources(
    sources
):

    if not sources:

        return

    with st.expander(
        "📚 Sources"
    ):

        for index, source in enumerate(
            sources,
            1,
        ):

            st.write(

                f"[{index}] "
                f"{source['source']} "
                f"— page/slide "
                f"{source['page']}"
            )


# ============================================================
# 20. INDEX UPLOADED FILES
# ============================================================

def index_uploaded_files(
    uploaded_files
):

    if not uploaded_files:

        return 0

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = []

    # --------------------------------------------------------
    # Save uploaded files
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Extract documents
    # --------------------------------------------------------

    documents: List[
        Document
    ] = []

    for path in paths:

        documents.extend(
            load_file(path)
        )

    if not documents:

        return 0

    # --------------------------------------------------------
    # Store in PGVector
    # --------------------------------------------------------

    get_vectorstore().add_documents(
        documents
    )

    return len(
        documents
    )


# ============================================================
# 21. MAIN STREAMLIT APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Page configuration
    # --------------------------------------------------------

    st.set_page_config(

        page_title=
            "MLF Fisheries Agentic AI",

        page_icon="🐟",

        layout="wide",
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.title(
        "🐟 MLF Fisheries Regulatory "
        "Compliance — Agentic AI"
    )

    st.caption(
        "LangGraph + PostgreSQL/pgvector "
         "+ Ollama + BGE-M3"
    )

    # --------------------------------------------------------
    # Conversation state
    # --------------------------------------------------------

    if (
        "thread_id"
        not in st.session_state
    ):

        st.session_state.thread_id = (
            f"user-{int(time.time())}"
        )

    if (
        "messages"
        not in st.session_state
    ):

        st.session_state.messages = []

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.subheader(
            "⚙️ Configuration"
        )

        st.write(
             f"Model: `{OLLAMA_MODEL}`"
        )

        st.write(
            f"Embedding: "
            f"`{EMBEDDING_MODEL}`"
        )

        st.write(
            f"Top-K: `{TOP_K}`"
        )

        st.write(
            "Database: "
            "`PostgreSQL + pgvector`"
        )

        st.write(
            "Database port: `5433`"
        )

        # ----------------------------------------------------
        # New conversation
        # ----------------------------------------------------

        if st.button(
            "🧹 New conversation"
        ):

            st.session_state.thread_id = (
                f"user-{int(time.time())}"
            )

            st.session_state.messages = []

            st.rerun()

        st.divider()

        # ====================================================
        # KNOWLEDGE BASE
        # ====================================================

        st.subheader(
            "📂 Knowledge Base"
        )

        st.caption(
            "Upload supported documents "
            "and images. Text is extracted "
            "and indexed into pgvector."
        )

        uploaded_files = (
            st.file_uploader(

                "Upload files",

                type=[
                    ext.lstrip(".")
                    for ext in
                    sorted(
                        SUPPORTED_EXTENSIONS
                    )
                ],

                accept_multiple_files=True,
            )
        )

        if st.button(

            "📥 Index uploaded files",

            disabled=not uploaded_files,
        ):

            with st.spinner(
                "Extracting text, "
                "creating embeddings "
                "and indexing…"
            ):

                try:

                    count = (
                        index_uploaded_files(
                            uploaded_files
                        )
                    )

                    st.success(

                        f"Indexed {count} "
                        "text chunks into "
                        "the knowledge base."
                    )

                except Exception as exc:

                    st.error(
                        f"Indexing failed: "
                        f"{exc}"
                    )

    # ========================================================
    # DISPLAY PREVIOUS MESSAGES
    # ========================================================

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

    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask in English or Swahili…"
    )

    if not question:

        return

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": question,
    })

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # ========================================================
    # ASSISTANT RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🤖 Agent is reasoning, "
            "retrieving and verifying…"
        ):

            start = time.perf_counter()

            try:

                result = (
                    get_graph().invoke(

                        {
                            "question":
                                question,

                            "attempts":
                                0,
                        },

                        config={
                            "configurable": {
                                "thread_id":
                                    st.session_state
                                    .thread_id
                            }
                        },
                    )
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

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.markdown(
            result["answer"]
        )

        # ----------------------------------------------------
        # Performance information
        # ----------------------------------------------------

        st.caption(

            f"⚡ {elapsed:.2f}s | "
            f"Retrieval relevance: "
            f"{result.get('relevance', 0):.2f} | "
            f"Language: "
            f"{result.get('language', 'en').upper()}"
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        render_sources(
            result.get(
                "sources",
                []
            )
        )

        # ----------------------------------------------------
        # Save conversation
        # ----------------------------------------------------

        st.session_state.messages.append({

            "role":
                "assistant",

            "content":
                result["answer"],

            "sources":
                result.get(
                    "sources",
                    []
                ),
        })


# ============================================================
# 22. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()