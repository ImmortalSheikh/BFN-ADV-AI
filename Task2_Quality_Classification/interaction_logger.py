import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "interactions.db"


def log_interaction(user_id, input_data, prediction):
    DB_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            input_data TEXT,
            prediction TEXT,
            timestamp TEXT
        )
    """)

    cursor.execute("""
        INSERT INTO logs (user_id, input_data, prediction, timestamp)
        VALUES (?, ?, ?, ?)
    """, (user_id, input_data, prediction, datetime.utcnow().isoformat()))

    conn.commit()
    conn.close()