import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

COLLECTION_NAME = "mlf_fisheries_regulations"

print("=" * 70)
print("MLF FISHERIES VECTOR RETRIEVAL TEST")
print("=" * 70)

print(f"Database: {DATABASE_URL}")
print(f"Collection: {COLLECTION_NAME}")

# ---------------------------------------------------------
# 1. Load the same embedding model used during ingestion
# ---------------------------------------------------------
print("\n[1/3] Loading BGE-M3 embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

print("Embedding model: OK")

# ---------------------------------------------------------
# 2. Connect to PostgreSQL + pgvector
# ---------------------------------------------------------
print("\n[2/3] Connecting to PostgreSQL + pgvector...")

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DATABASE_URL,
    use_jsonb=True,
)

print("Vector store: OK")

# ---------------------------------------------------------
# 3. Perform similarity search
# ---------------------------------------------------------
print("\n[3/3] Performing similarity search...")

query = "What are the regulations concerning fishing nets?"

results = vectorstore.similarity_search_with_score(
    query,
    k=5
)

print(f"\nQuery: {query}")
print(f"Results returned: {len(results)}")

print("\n" + "=" * 70)
print("RETRIEVED DOCUMENTS")
print("=" * 70)

for i, (doc, score) in enumerate(results, start=1):

    print(f"\n--- RESULT {i} ---")
    print(f"Similarity score: {score}")

    source = doc.metadata.get("source", "Unknown")
    print(f"Source: {source}")

    print("\nContent:")
    print(doc.page_content[:1000])

print("\n" + "=" * 70)
print("RETRIEVAL TEST COMPLETED")
print("=" * 70)