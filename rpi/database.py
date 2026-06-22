import sqlite3
from datetime import datetime

DB_PATH = "database/patients.db"

def init_db():
import os
os.makedirs("database", exist_ok=True)

```
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS patient_profile (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age TEXT,
        gender TEXT,
        weight TEXT,
        height TEXT,
        allergies TEXT,
        medications TEXT,
        history TEXT
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS doctor_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        assessment TEXT,
        plan TEXT
    )
""")

cur.execute("""
    INSERT OR IGNORE INTO patient_profile
    (id,name,age,gender,weight,height,allergies,medications,history)
    VALUES
    (1,'Demo Patient','45','N/A','70 kg','170 cm',
    'None','None','No history')
""")

conn.commit()
conn.close()
```

def get_profile():
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("SELECT * FROM patient_profile WHERE id=1")
row = cur.fetchone()
conn.close()
return dict(row) if row else {}

def save_note(data):
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

```
cur.execute("""
    INSERT INTO doctor_notes
    (timestamp,assessment,plan)
    VALUES (?,?,?)
""", (
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    data.get("assessment",""),
    data.get("plan","")
))

conn.commit()
conn.close()
```

def get_notes():
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

```
cur.execute("""
    SELECT * FROM doctor_notes
    ORDER BY id DESC
    LIMIT 20
""")

rows = cur.fetchall()
conn.close()

return [dict(r) for r in rows]
```
