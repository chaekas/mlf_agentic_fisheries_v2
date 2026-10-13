import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load database settings from your .env file
load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    print("ERROR: DATABASE_URL was not found in your .env file.")
    raise SystemExit(1)

try:
    engine = create_engine(database_url, pool_pre_ping=True)

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT username, role, created_at
                FROM mlf_user_accounts
                WHERE LOWER(username) = LOWER(:username)
            """),
            {"username": "chaekas"},
        )

        user = result.fetchone()

        if user:
            print("ACCOUNT FOUND")
            print(f"Username: {user.username}")
            print(f"Role: {user.role}")
            print(f"Created at: {user.created_at}")
        else:
            print("ACCOUNT NOT FOUND in this database.")

except Exception as error:
    print(f"Database check failed: {error}")

finally:
    if "engine" in locals():
        engine.dispose()
