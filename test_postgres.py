import psycopg

conn = psycopg.connect(
    "postgresql://postgres:postgres@localhost:5433/postgres"
)

cur = conn.cursor()

cur.execute(
    "SELECT extname FROM pg_extension WHERE extname = 'vector'"
)

print("POSTGRESQL: CONNECTED")
print("PGVECTOR:", cur.fetchall())

cur.close()
conn.close()