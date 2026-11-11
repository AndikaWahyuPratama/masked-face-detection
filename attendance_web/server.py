"""Module: server.py
Fungsi: Server Web Backend Ringan (Flask REST API + Native MJPEG Video Streaming)
Fitur Utama:
  1. Real-time Video Stream (MJPEG 30+ FPS, Super Enteng, No-Lag)
  2. Face ID Multi-Shot Registration API (Centroid Embedding ala Smartphone)
  3. Real-time Attendance & Mask Compliance Logging ke SQLite
  4. User Management & Analytics API
"""

import os
import sys
import time
import base64
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, request, jsonify, send_from_directory

# Pastikan path import terarah ke attendance_web
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import db_manager
from face_engine import FaceEngine

app = Flask(__name__, 
            template_folder=str(CURRENT_DIR / "templates"),
            static_folder=str(CURRENT_DIR / "static"))

# Inisialisasi Database
db_manager.init_db()

# Inisialisasi AI Engine
print("[Server] Menginisialisasi FaceEngine (YOLOv8 + FaceNet PyTorch)...")
engine = FaceEngine()

# Cache Pengguna Terdaftar (untuk performa kencang tanpa query DB tiap frame)
cached_users_lock = threading.Lock()
cached_users: List[Dict[str, Any]] = []

def reload_cached_users():
    global cached_users
    with cached_users_lock:
        cached_users = db_manager.get_all_users()
        print(f"[Server] Cache pengguna diperbarui: {len(cached_users)} anggota terdaftar.")

reload_cached_users()

# State Terakhir Deteksi (untuk notifikasi live HUD di UI)
latest_detection_lock = threading.Lock()
latest_detection: Dict[str, Any] = {
    "detected": False,
    "name": "Menunggu Wajah...",
    "identifier": "",
    "mask_status": "",
    "mask_label": "",
    "compliance": "",
    "confidence": 0.0,
    "timestamp": ""
}

# ==========================================
# CAMERA THREAD (Zero Latency Video Capture)
# ==========================================
class CameraStream:
    def __init__(self, src: int = 0, width: int = 640, height: int = 480):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.is_running = True
        self.lock = threading.Lock()
        self.fps = 0.0
        self.thread = threading.Thread(target=self._capture_thread_worker, daemon=True)
        self.thread.start()

    def _open_camera(self):
        """Membuka kamera webcam default."""
        try:
            cap = cv2.VideoCapture(self.src)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"[CameraStream] Kamera {self.src} berhasil dihubungkan.")
                return cap
        except Exception as e:
            print(f"[CameraStream] Error membuka kamera: {e}")
        return None

    def _capture_thread_worker(self):
        """Thread background terus membaca frame agar buffer kamera selalu terupdate (0 latency)."""
        prev_time = time.time()
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open_camera()
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(1.0) # Coba lagi 1 detik kemudian
                    continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                curr_time = time.time()
                dt = curr_time - prev_time
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                prev_time = curr_time
            else:
                time.sleep(0.01)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def stop(self):
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("[CameraStream] Kamera dimatikan.")

# Instansiasi Kamera Global
camera = CameraStream(src=0, width=640, height=480)

# ==========================================
# MJPEG STREAM GENERATOR
# ==========================================
def generate_mjpeg_stream():
    """Generator streaming video MJPEG native ke browser."""
    global latest_detection
    
    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            # Placeholder jika kamera belum aktif/tersedia
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Menghubungkan ke Kamera...", (140, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            ret, jpeg = cv2.imencode('.jpg', placeholder)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
            
        # Ambil daftar pengguna dari cache
        with cached_users_lock:
            active_users = list(cached_users)
            
        # Proses AI Frame (YOLOv8 MFD + FaceNet)
        # Threshold dikunci optimal: conf=0.25, distance_threshold=0.48
        annotated_frame, events = engine.process_frame(
            frame, 
            active_users, 
            conf_threshold=0.25, 
            distance_threshold=0.48
        )
        
        # Proses event deteksi & pencatatan absensi otomatis
        if events:
            # Ambil event wajah terdepan/terjelas
            primary_event = events[0]
            name = primary_event["name"]
            identifier = primary_event["identifier"]
            mask_status = primary_event["mask_status"]
            mask_label = primary_event["mask_label"]
            compliance = primary_event["compliance"]
            conf = primary_event["confidence"]
            user_id = primary_event["user_id"]
            crop_img = primary_event["crop"]
            
            # Update state terbaru untuk HUD realtime
            with latest_detection_lock:
                latest_detection = {
                    "detected": True,
                    "name": name,
                    "identifier": identifier,
                    "mask_status": mask_status,
                    "mask_label": mask_label,
                    "compliance": compliance,
                    "confidence": round(conf * 100, 1),
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                
            # Log Presensi Otomatis jika terdaftar dan belum absen dalam cooldown
            if identifier and identifier != "Unknown" and user_id is not None:
                if db_manager.can_attend(identifier, cooldown_minutes=5):
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_name = f"log_{identifier}_{ts_str}.jpg"
                    snap_path = db_manager.SNAPSHOTS_DIR / snap_name
                    cv2.imwrite(str(snap_path), crop_img)
                    
                    db_manager.record_attendance(
                        name=name,
                        identifier=identifier,
                        mask_status=mask_status,
                        compliance=compliance,
                        confidence=conf,
                        snapshot_path=str(snap_path),
                        user_id=user_id
                    )
                    print(f"[ATTENDANCE SUCCESS] {name} ({identifier}) - {compliance} ({mask_label})")
        else:
            with latest_detection_lock:
                latest_detection["detected"] = False
                
        # Encode frame ke JPEG untuk dikirim via MJPEG
        ret, jpeg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                   
        # Kontrol jeda agar smooth (target 30 FPS)
        time.sleep(0.015)

# ==========================================
# ROUTES & REST APIS
# ==========================================

@app.route("/")
def index():
    """Halaman Dashboard Utama."""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Endpoint Streaming Video Native MJPEG."""
    return Response(generate_mjpeg_stream(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/status")
def api_status():
    """Status sistem & hardware akselerasi."""
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    return jsonify({
        "status": "online",
        "device": "CUDA GPU" if cuda_avail else "CPU",
        "gpu_name": gpu_name,
        "camera_active": camera.is_running,
        "camera_fps": round(camera.fps, 1),
        "total_users": len(cached_users)
    })

@app.route("/api/latest_event")
def api_latest_event():
    """Event deteksi wajah & masker paling aktual untuk UI polling."""
    with latest_detection_lock:
        return jsonify(latest_detection)

@app.route("/api/stats")
def api_stats():
    """Statistik presensi & kepatuhan masker hari ini."""
    summary = db_manager.get_today_summary()
    return jsonify(summary)

@app.route("/api/attendance_today")
def api_attendance_today():
    """Daftar log presensi hari ini."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    compliance = request.args.get("compliance", "Semua")
    logs = db_manager.get_attendance_logs(date_filter=today_str, compliance_filter=compliance, limit=100)
    
    # Format agar path snapshot bisa diakses via web
    for log in logs:
        if log.get("snapshot_path"):
            fname = Path(log["snapshot_path"]).name
            log["snapshot_url"] = f"/snapshots/{fname}"
        else:
            log["snapshot_url"] = None
            
    return jsonify(logs)

@app.route("/api/users", methods=["GET"])
def api_get_users():
    """Daftar seluruh pengguna terdaftar."""
    with cached_users_lock:
        users_list = []
        for u in cached_users:
            u_clean = dict(u)
            u_clean.pop("embedding", None) # Jangan kirim embedding besar ke frontend
            if u_clean.get("photo_path"):
                u_clean["photo_url"] = f"/registered_faces/{Path(u_clean['photo_path']).name}"
            users_list.append(u_clean)
        return jsonify(users_list)

@app.route("/api/users/<int:user_id>", methods=["DELETE"])
def api_delete_user(user_id: int):
    """Menghapus pengguna dari database."""
    success, msg = db_manager.delete_user(user_id)
    if success:
        reload_cached_users()
        return jsonify({"success": True, "message": msg})
    return jsonify({"success": False, "message": msg}), 400

@app.route("/api/register_face_id", methods=["POST"])
def api_register_face_id():
    """Pendaftaran Pengguna dengan Multi-Shot Face ID (Ala Smartphone).
    Menerima:
      {
        "name": "Lukman",
        "identifier": "4152401009",
        "role": "Mahasiswa",
        "images": ["data:image/jpeg;base64,...", "data:image/jpeg;base64,...", ...]
      }
    """
    try:
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        identifier = data.get("identifier", "").strip()
        role = data.get("role", "Mahasiswa").strip()
        images_b64 = data.get("images", [])

        if not name or not identifier:
            return jsonify({"success": False, "message": "Nama dan NIM/ID wajib diisi!"}), 400

        if not images_b64 or len(images_b64) < 1:
            return jsonify({"success": False, "message": "Minimal harus mengambil 1 sudut foto wajah!"}), 400

        # Decode gambar base64 ke format numpy BGR
        decoded_images = []
        for idx, b64_str in enumerate(images_b64):
            try:
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    decoded_images.append(img)
            except Exception as e:
                print(f"[Face ID Decode Error] Frame {idx}: {e}")

        if not decoded_images:
            return jsonify({"success": False, "message": "Gagal membaca format gambar wajah."}), 400

        # Ekstraksi Multi-Shot Centroid Embedding
        success, centroid_emb, best_crop, msg = engine.enroll_face_id_multishot(decoded_images)
        if not success or centroid_emb is None:
            return jsonify({"success": False, "message": msg}), 400

        # Simpan foto profil terbaik
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = "".join([c for c in identifier if c.isalnum() or c in ("-", "_")])
        photo_filename = f"{safe_id}_{ts_str}.jpg"
        photo_path = db_manager.REGISTERED_FACES_DIR / photo_filename
        
        save_img = best_crop if best_crop is not None else decoded_images[0]
        cv2.imwrite(str(photo_path), save_img)

        # Simpan ke Database SQLite
        ok, db_msg = db_manager.add_user(
            name=name,
            identifier=identifier,
            role=role,
            photo_path=str(photo_path),
            embedding=centroid_emb
        )

        if ok:
            reload_cached_users()
            return jsonify({
                "success": True, 
                "message": f"Face ID berhasil didaftarkan untuk {name} ({len(decoded_images)} sudut terkalibrasi)."
            })
        else:
            return jsonify({"success": False, "message": db_msg}), 400

    except Exception as e:
        print(f"[ERROR api_register_face_id] {e}")
        return jsonify({"success": False, "message": f"Terjadi kesalahan server: {str(e)}"}), 500

# Endpoint static untuk menyajikan gambar snapshot & foto wajah
@app.route("/snapshots/<filename>")
def serve_snapshot(filename):
    return send_from_directory(str(db_manager.SNAPSHOTS_DIR), filename)

@app.route("/registered_faces/<filename>")
def serve_face(filename):
    return send_from_directory(str(db_manager.REGISTERED_FACES_DIR), filename)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 SISTEM ABSENSI CERDAS & MASKED FACE DETECTION (MFD)")
    print("   Framework: Flask + Vanilla JS/HTML5 (Super Enteng & Smooth)")
    print("   Akses Web di browser Anda: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
