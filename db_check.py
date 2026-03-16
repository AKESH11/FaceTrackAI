import sqlite3
from datetime import date

db_path = "instance/attendance.db"
con = sqlite3.connect(db_path)
cur = con.cursor()

print("Using DB:", db_path)
print("\nTable counts:")
tables = [
    "users",
    "assigned_classes",
    "students",
    "face_encodings",
    "timetables",
    "attendances",
    "pending_confirmations",
    "settings",
]
for t in tables:
    n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"{t}: {n}")

print("\nLatest 15 attendance rows:")
rows = cur.execute("""
SELECT a.id, s.name, s.roll_number, a.date, a.time_marked, a.subject, a.status, a.confidence
FROM attendances a
LEFT JOIN students s ON s.id = a.student_id
ORDER BY a.id DESC
LIMIT 15
""").fetchall()
for r in rows:
    print(r)

today = str(date.today())
print("\nToday's attendance count:", cur.execute(
    "SELECT COUNT(*) FROM attendances WHERE date = ?", (today,)
).fetchone()[0])

con.close()