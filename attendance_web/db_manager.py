"""Module: db_manager.py
Fungsi: Manajemen Database SQLite untuk Sistem Absensi Cerdas & Kepatuhan Masker
Lokasi Database: attendance_web/attendance.db
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "attendance.db"
REGISTERED_FACES_DIR = BASE_DIR / "registered_faces"
SNAPSHOTS_DIR = BASE_DIR / "logs_snapshots"

def get_connection() -> sqlite3.Connection:
    """Membuka koneksi ke SQLite dengan row_factory dictionary."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Inisialisasi tabel users dan attendance_logs serta folder penyimpanan."""
    REGISTERED_FACES_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = get_connection()
    c = conn.cursor()
    
    # 1. Tabel users (Data anggota terdaftar)
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        identifier TEXT UNIQUE NOT NULL,
        role TEXT DEFAULT 'Mahasiswa',
        photo_path TEXT NOT NULL,
        embedding TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 2. Tabel attendance_logs (Riwayat presensi & kepatuhan masker)
    c.execute("""
    CREATE TABLE IF NOT EXISTS attendance_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT NOT NULL,
        identifier TEXT,
        mask_status TEXT NOT NULL,
        compliance TEXT NOT NULL,
        confidence REAL,
        snapshot_path TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
    )
    """)
    
    conn.commit()
    conn.close()

def add_user(name: str, identifier: str, role: str, photo_path: str, embedding: Optional[List[float]] = None) -> Tuple[bool, str]:
    """Mendaftarkan anggota baru beserta lokasi foto dan embedding vektor wajah."""
    conn = get_connection()
    c = conn.cursor()
    try:
        emb_json = json.dumps(embedding) if embedding is not None else None
        c.execute("""
        INSERT INTO users (name, identifier, role, photo_path, embedding)
        VALUES (?, ?, ?, ?, ?)
        """, (name.strip(), identifier.strip(), role.strip(), str(photo_path), emb_json))
        conn.commit()
        return True, f"Anggota '{name}' (ID: {identifier}) berhasil didaftarkan."
    except sqlite3.IntegrityError:
        return False, f"NIM/ID '{identifier}' sudah terdaftar dalam sistem!"
    except Exception as e:
        return False, f"Terjadi kesalahan database: {str(e)}"
    finally:
        conn.close()

def update_user_embedding(user_id: int, embedding: List[float]) -> bool:
    """Memperbarui nilai embedding wajah untuk user tertentu."""
    conn = get_connection()
    c = conn.cursor()
    try:
        emb_json = json.dumps(embedding)
        c.execute("UPDATE users SET embedding = ? WHERE id = ?", (emb_json, user_id))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def get_all_users() -> List[Dict[str, Any]]:
    """Mengambil semua profil anggota terdaftar."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, identifier, role, photo_path, embedding, created_at FROM users ORDER BY name ASC")
    rows = c.fetchall()
    conn.close()
    
    users = []
    for r in rows:
        item = dict(r)
        if item.get("embedding"):
            try:
                item["embedding"] = json.loads(item["embedding"])
            except Exception:
                item["embedding"] = None
        users.append(item)
    return users

def delete_user(user_id: int) -> Tuple[bool, str]:
    """Menghapus data anggota berdasarkan ID."""
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT photo_path FROM users WHERE id = ?", (user_id,))
        row = c.fetchone()
        if row and row["photo_path"]:
            p = Path(row["photo_path"])
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return True, "Anggota berhasil dihapus."
    except Exception as e:
        return False, f"Gagal menghapus anggota: {str(e)}"
    finally:
        conn.close()

def can_attend(identifier: str, cooldown_minutes: int = 5) -> bool:
    """Mengecek apakah orang dengan NIM/Identifier ini sudah presensi dalam interval cooldown.
    Mencegah pencatatan berulang (spamming log) dalam durasi tertentu.
    """
    if not identifier or identifier.lower() == "unknown":
        return False # Jangan catat spam jika orang tidak dikenal
    
    conn = get_connection()
    c = conn.cursor()
    threshold = (datetime.now() - timedelta(minutes=cooldown_minutes)).strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
    SELECT id FROM attendance_logs 
    WHERE identifier = ? AND timestamp >= ?
    LIMIT 1
    """, (identifier, threshold))
    res = c.fetchone()
    conn.close()
    return res is None

def record_attendance(name: str, identifier: Optional[str], mask_status: str, 
                      compliance: str, confidence: float, snapshot_path: Optional[str] = None, 
                      user_id: Optional[int] = None) -> bool:
    """Mencatat presensi baru ke tabel attendance_logs dengan waktu lokal konsisten."""
    conn = get_connection()
    c = conn.cursor()
    try:
        now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("""
        INSERT INTO attendance_logs (user_id, name, identifier, mask_status, compliance, confidence, snapshot_path, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, name, identifier, mask_status, compliance, float(confidence), snapshot_path, now_local))
        conn.commit()
        return True
    except Exception as e:
        print(f"[ERROR record_attendance] {e}")
        return False
    finally:
        conn.close()

def get_attendance_logs(date_filter: Optional[str] = None, compliance_filter: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
    """Mengambil riwayat absensi dengan filter opsional tanggal dan kepatuhan masker."""
    conn = get_connection()
    c = conn.cursor()
    
    query = "SELECT id, user_id, name, identifier, mask_status, compliance, confidence, snapshot_path, timestamp FROM attendance_logs WHERE 1=1"
    params = []
    
    if date_filter:
        query += " AND date(timestamp) = date(?)"
        params.append(date_filter)
        
    if compliance_filter and compliance_filter != "Semua":
        query += " AND compliance = ?"
        params.append(compliance_filter)
        
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    c.execute(query, tuple(params))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_today_summary() -> Dict[str, Any]:
    """Menghitung metrik ringkasan kehadiran hari ini: total hadir, persentase patuh, melanggar."""
    conn = get_connection()
    c = conn.cursor()
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # Hitung unik orang yang hadir hari ini
    c.execute("""
    SELECT 
        COUNT(DISTINCT CASE WHEN identifier IS NOT NULL AND identifier != 'Unknown' THEN identifier ELSE name END) as total_present,
        SUM(CASE WHEN compliance = 'PATUH' THEN 1 ELSE 0 END) as total_patuh,
        SUM(CASE WHEN compliance = 'MELANGGAR' THEN 1 ELSE 0 END) as total_melanggar,
        COUNT(*) as total_logs
    FROM attendance_logs 
    WHERE date(timestamp) = date(?)
    """, (today_str,))
    
    res = c.fetchone()
    conn.close()
    
    total_present = res["total_present"] if res else 0
    total_patuh = res["total_patuh"] if res and res["total_patuh"] else 0
    total_melanggar = res["total_melanggar"] if res and res["total_melanggar"] else 0
    total_logs = res["total_logs"] if res and res["total_logs"] else 0
    
    patuh_pct = (total_patuh / total_logs * 100.0) if total_logs > 0 else 100.0
    
    return {
        "total_present": total_present,
        "total_patuh": total_patuh,
        "total_melanggar": total_melanggar,
        "total_logs": total_logs,
        "patuh_pct": round(patuh_pct, 1)
    }

if __name__ == "__main__":
    init_db()
    print("Database SQLite & folder penyimpanan berhasil diinisialisasi!")

