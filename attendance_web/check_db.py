import sqlite3
import json
from pathlib import Path

db_path = Path("attendance_web/attendance.db")
if not db_path.exists():
    print("[ERROR] Database attendance.db belum ditemukan.")
    exit()

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
c = conn.cursor()

c.execute("SELECT id, name, identifier, role, photo_path, embedding, created_at FROM users")
users = c.fetchall()

print(f"\n================ TOTAL ANGGOTA TERDAFTAR: {len(users)} ORANG ================")
for u in users:
    emb = json.loads(u["embedding"]) if u["embedding"] else None
    emb_len = len(emb) if emb else 0
    p = Path(u["photo_path"])
    print(f"ID Database     : {u['id']}")
    print(f"Nama Lengkap    : {u['name']}")
    print(f"NIM / NIP       : {u['identifier']}")
    print(f"Peran / Role    : {u['role']}")
    print(f"File Pasfoto    : {u['photo_path']} (Tersimpan Fisik: {p.exists()})")
    print(f"Vektor AI       : {emb_len} Dimensi (FaceNet Biometrics)")
    print(f"Waktu Mendaftar : {u['created_at']}")
    print("-" * 65)

c.execute("SELECT id, name, identifier, mask_status, compliance, confidence, timestamp FROM attendance_logs ORDER BY id DESC LIMIT 5")
logs = c.fetchall()
print(f"\n================ RIWAYAT PRESENSI: {len(logs)} LOG TERBARU ================")
for l in logs:
    print(f"Log #{l['id']} | {l['timestamp']} | {l['name']} ({l['identifier']}) | Status: {l['mask_status']} | Kepatuhan: {l['compliance']} | AI: {l['confidence']*100:.1f}%")

conn.close()

