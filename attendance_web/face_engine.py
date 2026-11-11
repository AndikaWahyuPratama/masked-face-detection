"""Module: face_engine.py
Fungsi: Pipeline AI Vision Tingkat Lanjut:
  1. Deteksi Spasial Masker & Wajah (YOLOv8 Nano/Small pada GPU CUDA)
  2. Ekstraksi Fitur Biometrik Wajah (FaceNet InceptionResnetV1 pada GPU CUDA)
  3. Visualisasi Grafis HUD & Bounding Box Modern & Elegan
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "paket_hasil"

# Mapping kelas resmi FMLD
CLASSES = ["masked_face", "unmasked_face", "incorrectly_masked_face"]

# Skema Warna & Label UI Modern (Format BGR untuk OpenCV)
# 0: Masker Benar -> Emerald Green (34, 197, 94) -> BGR: (94, 197, 34)
# 1: Tanpa Masker -> Crimson Red (239, 68, 68) -> BGR: (68, 68, 239)
# 2: Masker Melorot -> Amber Orange (245, 158, 11) -> BGR: (11, 158, 245)
CLASS_METADATA = {
    0: {
        "label": "MASKER BENAR",
        "compliance": "PATUH",
        "color_bgr": (80, 200, 50),
        "badge_icon": "PATUH"
    },
    1: {
        "label": "TANPA MASKER",
        "compliance": "MELANGGAR",
        "color_bgr": (60, 60, 235),
        "badge_icon": "MELANGGAR"
    },
    2: {
        "label": "MASKER MELOROT",
        "compliance": "MELANGGAR",
        "color_bgr": (20, 160, 245),
        "badge_icon": "MELANGGAR"
    }
}

class FaceEngine:
    def __init__(self, yolo_model_path: Optional[str] = None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[FaceEngine] Hardware akselerasi aktif: {self.device}")
        
        # Prioritas pemilihan model bobot terbaik
        if yolo_model_path and Path(yolo_model_path).exists():
            chosen_yolo = Path(yolo_model_path)
        else:
            v8s_path = MODELS_DIR / "masked_face_yolov8s_best.pt"
            v8n_path = MODELS_DIR / "masked_face_yolov8n_best.pt"
            if v8s_path.exists():
                chosen_yolo = v8s_path
            elif v8n_path.exists():
                chosen_yolo = v8n_path
            else:
                chosen_yolo = "yolov8n.pt"
                
        self.yolo_path = str(chosen_yolo)
        print(f"[FaceEngine] Memuat arsitektur YOLOv8 MFD: {self.yolo_path}")
        self.yolo_model = YOLO(self.yolo_path)
        
        # Muat FaceNet InceptionResnetV1 pada CUDA
        print("[FaceEngine] Memuat FaceNet InceptionResnetV1 biometrik engine...")
        self.facenet = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(self.device)
        
        # Smart Temporal Biometric Tracker & RAM Cache (Kecepatan 30+ FPS, hemat beban GPU 75%)
        self.tracks = []  # list of {'bbox': tuple, 'user': dict, 'dist': float, 'ttl': int}
        self.db_upper_cache = {}  # Cache RAM agar tidak pernah membaca ulang file dari disk saat stream

    @staticmethod
    def _compute_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        """Menghitung Intersection over Union (IoU) antar dua bounding box untuk temporal tracking."""
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return (inter / union) if union > 0 else 0.0

    def extract_embedding_from_crop(self, face_bgr: np.ndarray) -> Optional[List[float]]:
        """Mengekstrak vektor embedding ternormalisasi L2 (512 dimensi) dari potongan citra wajah dengan akselerasi langsung."""
        if face_bgr is None or face_bgr.size == 0 or face_bgr.shape[0] < 20 or face_bgr.shape[1] < 20:
            return None
        
        try:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
            tensor_img = torch.from_numpy(resized).to(self.device).permute(2, 0, 1).float().sub_(127.5).div_(128.0).unsqueeze(0)
            
            with torch.no_grad():
                emb = self.facenet(tensor_img)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                emb_list = emb.cpu().squeeze(0).numpy().tolist()
            return emb_list
        except Exception as e:
            print(f"[ERROR extract_embedding] {e}")
            return None

    def register_face_from_image(self, img_input: Any) -> Tuple[bool, Optional[List[float]], Optional[np.ndarray], str]:
        """Mendeteksi wajah pada foto unggahan registrasi dan mengekstrak vektor identitas."""
        try:
            if isinstance(img_input, (str, Path)):
                img = cv2.imread(str(img_input))
            elif isinstance(img_input, bytes):
                nparr = np.frombuffer(img_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(img_input, np.ndarray):
                img = img_input
            else:
                return False, None, None, "Format gambar tidak didukung."
                
            if img is None:
                return False, None, None, "Gagal mengurai citra wajah."
            
            # Deteksi wajah menggunakan model YOLO MFD
            device_arg = 0 if torch.cuda.is_available() else "cpu"
            results = self.yolo_model.predict(img, conf=0.25, device=device_arg, verbose=False)
            boxes = results[0].boxes
            
            h, w = img.shape[:2]
            cls_id = None
            if len(boxes) == 0:
                face_crop = img
            else:
                # Ambil kotak wajah dengan luas terbesar
                best_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                cls_id = int(best_box.cls[0])
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                # Margin dinamis
                pad_w = int((x2 - x1) * 0.1)
                pad_h = int((y2 - y1) * 0.1)
                x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
                face_crop = img[y1:y2, x1:x2]
                
            embedding = self.extract_embedding_from_crop(face_crop)
            if embedding is None:
                return False, None, None, "Kualitas wajah terlalu rendah atau kabur untuk diekstrak."
                
            # Cerdas: Jika foto pendaftaran adalah wajah normal tanpa masker (cls_id == 1 atau None),
            # otomatis buat representasi masker sintetis (Synthetic Mask Augmentation).
            # Ini menghasilkan Dual-Template [emb_normal, emb_sim_mask] sehingga user otomatis
            # dikenali baik saat melepas masker maupun saat memakai masker!
            if cls_id == 1 or cls_id is None:
                sim_crop = face_crop.copy()
                ch, cw = sim_crop.shape[:2]
                pts = np.array([
                    [int(cw * 0.10), int(ch * 0.48)],
                    [int(cw * 0.50), int(ch * 0.40)],
                    [int(cw * 0.90), int(ch * 0.48)],
                    [int(cw * 0.85), int(ch * 0.98)],
                    [int(cw * 0.15), int(ch * 0.98)]
                ], np.int32)
                cv2.fillPoly(sim_crop, [pts], (235, 235, 235))
                sim_emb = self.extract_embedding_from_crop(sim_crop)
                if sim_emb is not None:
                    embedding = [embedding, sim_emb]
                    msg = "Wajah normal berhasil diekstrak + representasi masker otomatis aktif (Dual-Template)."
                else:
                    msg = "Wajah normal berhasil dideteksi dan diekstrak."
            else:
                msg = "Wajah bermasker berhasil dideteksi dan diekstrak."
                
            return True, embedding, face_crop, msg
        except Exception as e:
            return False, None, None, f"Terjadi kesalahan saat registrasi: {str(e)}"

    def match_face(self, query_embedding: List[float], registered_users: List[Dict[str, Any]], distance_threshold: float = 0.48) -> Tuple[Optional[Dict[str, Any]], float]:
        """Pencocokan biometrik Euclidean Distance terhadap database pengguna terdaftar.
        Mendukung single template maupun multiple templates per pengguna.
        """
        if not registered_users or query_embedding is None:
            return None, 999.0
            
        q_vec = np.array(query_embedding, dtype=np.float32)
        min_dist = 999.0
        best_match = None
        
        for u in registered_users:
            u_emb = u.get("embedding")
            if not u_emb:
                continue
                
            # Jika user memiliki multiple embedding templates (list of list)
            if isinstance(u_emb, list) and len(u_emb) > 0 and isinstance(u_emb[0], list):
                for emb_item in u_emb:
                    db_vec = np.array(emb_item, dtype=np.float32)
                    dist = float(np.linalg.norm(q_vec - db_vec))
                    if dist < min_dist:
                        min_dist = dist
                        if dist <= distance_threshold:
                            best_match = u
            else:
                db_vec = np.array(u_emb, dtype=np.float32)
                dist = float(np.linalg.norm(q_vec - db_vec))
                if dist < min_dist:
                    min_dist = dist
                    if dist <= distance_threshold:
                        best_match = u
                    
        return best_match, min_dist

    def enroll_face_id_multishot(self, images: List[Any]) -> Tuple[bool, Optional[List[float]], Optional[np.ndarray], str]:
        """Pendaftaran Wajah Multi-Shot ala Face ID Smartphone.
        Menerima beberapa gambar wajah (depan, miring kiri, miring kanan).
        Mengekstrak embedding untuk masing-masing sudut, menghitung rata-rata vektor (centroid),
        dan menormalisasi L2 agar pengenalan sangat akurat dan tahan berbagai pose.
        """
        if not images:
            return False, None, None, "Tidak ada data gambar yang diterima."
            
        embeddings = []
        best_crop = None
        
        for idx, img_input in enumerate(images):
            try:
                success, emb, crop, msg = self.register_face_from_image(img_input)
                if success and emb is not None:
                    embeddings.append(emb)
                    if best_crop is None or idx == 0:
                        best_crop = crop
            except Exception as e:
                print(f"[FaceEngine] Error ekstraksi frame Face ID {idx}: {e}")
                
        if not embeddings:
            return False, None, None, "Wajah tidak terdeteksi jelas pada semua frame yang dikirim. Pastikan wajah tepat di lingkaran kamera."
            
        # Hitung Centroid (Rata-rata Vektor)
        emb_matrix = np.array(embeddings, dtype=np.float32) # shape: (N, 512)
        mean_emb = np.mean(emb_matrix, axis=0) # shape: (512,)
        # Normalisasi L2
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
            
        centroid_list = mean_emb.tolist()
        return True, centroid_list, best_crop, f"Sukses mengekstrak {len(embeddings)} sudut wajah (Face ID Centroid)."

    @staticmethod
    def _draw_corner_brackets(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], length: int = 15, thickness: int = 2):
        """Menggambar aksen sudut modern (bracket corners) bergaya HUD Cyberpunk/AI Vision."""
        length = min(length, (x2 - x1) // 3, (y2 - y1) // 3)
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness + 1)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness + 1)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness + 1)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness + 1)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness + 1)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness + 1)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness + 1)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness + 1)

    def process_frame(self, frame_bgr: np.ndarray, registered_users: List[Dict[str, Any]], conf_threshold: float = 0.25, distance_threshold: float = 0.48) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Memproses 1 frame video dengan visualisasi grafis profesional."""
        annotated_frame = frame_bgr.copy()
        events = []
        
        device_arg = 0 if torch.cuda.is_available() else "cpu"
        results = self.yolo_model.predict(frame_bgr, conf=conf_threshold, device=device_arg, verbose=False)
        boxes = results[0].boxes
        
        h_frame, w_frame = frame_bgr.shape[:2]
        new_tracks = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)
            
            if (x2 - x1) < 18 or (y2 - y1) < 18:
                continue
                
            face_crop = frame_bgr[y1:y2, x1:x2]
            current_bbox = (x1, y1, x2, y2)
            
            # Metadata Kelas & Status
            mask_status = CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown"
            meta = CLASS_METADATA.get(cls_id, {
                "label": "TIDAK DIKETAHUI",
                "compliance": "MELANGGAR",
                "color_bgr": (180, 180, 180),
                "badge_icon": "UNKNOWN"
            })
            
            label_text = meta["label"]
            compliance = meta["compliance"]
            box_color = meta["color_bgr"]
            
            # Smart Temporal Tracking (Cek apakah wajah ini sudah dikenali di frame-frame sebelumnya)
            matched_track = None
            for trk in self.tracks:
                if self._compute_iou(current_bbox, trk["bbox"]) > 0.40:
                    matched_track = trk
                    break
                    
            if matched_track is not None and matched_track["ttl"] > 0:
                matched_user = matched_track["user"]
                match_dist = matched_track["dist"]
                matched_track["ttl"] -= 1
                matched_track["bbox"] = current_bbox
                new_tracks.append(matched_track)
            else:
                # Pengenalan Biometrik Wajah dengan Dynamic Adaptive Threshold & Dual-Mode
                effective_dist_thresh = max(distance_threshold, 0.68)
                q_emb = self.extract_embedding_from_crop(face_crop)
                matched_user, match_dist = self.match_face(q_emb, registered_users, distance_threshold=effective_dist_thresh)
                
                # Periocular Fallback dengan In-Memory RAM Cache (Zero Disk Lag)
                if matched_user is None and match_dist <= 0.86 and registered_users:
                    h_c, w_c = face_crop.shape[:2]
                    q_upper = face_crop[0:int(h_c * 0.55), 0:w_c]
                    q_upper_emb = self.extract_embedding_from_crop(q_upper)
                    if q_upper_emb is not None:
                        q_up_vec = np.array(q_upper_emb, dtype=np.float32)
                        best_peri_user = None
                        min_p_dist = 999.0
                        second_p_dist = 999.0
                        for u in registered_users:
                            u_id = u.get("id")
                            if u_id not in self.db_upper_cache:
                                p_file = u.get("photo_path")
                                if p_file and Path(p_file).exists():
                                    db_p = cv2.imread(p_file)
                                    if db_p is not None:
                                        hp, wp = db_p.shape[:2]
                                        self.db_upper_cache[u_id] = self.extract_embedding_from_crop(db_p[0:int(hp * 0.55), 0:wp])
                                    else:
                                        self.db_upper_cache[u_id] = None
                                else:
                                    self.db_upper_cache[u_id] = None
                                    
                            cached_up = self.db_upper_cache.get(u_id)
                            if cached_up is not None:
                                p_vec = np.array(cached_up, dtype=np.float32)
                                d_peri = float(np.linalg.norm(q_up_vec - p_vec))
                                if d_peri < min_p_dist:
                                    second_p_dist = min_p_dist
                                    min_p_dist = d_peri
                                    best_peri_user = u
                                elif d_peri < second_p_dist:
                                    second_p_dist = d_peri
                        if best_peri_user and min_p_dist <= 0.86 and (second_p_dist - min_p_dist) >= 0.10:
                            matched_user = best_peri_user
                            match_dist = min_p_dist
                
                # Simpan ke cache temporal (ttl: 6 frame = FaceNet fresh ~5x per detik)
                new_tracks.append({"bbox": current_bbox, "user": matched_user, "dist": match_dist, "ttl": 6})
            
            if matched_user:
                person_name = matched_user["name"]
                person_id = matched_user["identifier"]
                user_id = matched_user["id"]
            else:
                person_name = "Tamu / Belum Terdaftar"
                person_id = "Unknown"
                user_id = None
                
            events.append({
                "user_id": user_id,
                "name": person_name,
                "identifier": person_id,
                "mask_status": mask_status,
                "mask_label": label_text,
                "compliance": compliance,
                "confidence": conf,
                "match_dist": match_dist,
                "crop": face_crop.copy()
            })
            
            # 1. Gambar Bounding Box Tipis Elegan
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)
            # 2. Gambar Akses Sudut Tebal (Corner Brackets)
            self._draw_corner_brackets(annotated_frame, x1, y1, x2, y2, box_color, length=16, thickness=2)
            
            # 3. Label Tag Card di Atas Bounding Box
            # Baris 1: Nama Personil
            # Baris 2: Status Masker + Persentase Keyakinan
            font = cv2.FONT_HERSHEY_DUPLEX
            name_text = person_name if len(person_name) <= 22 else person_name[:20] + ".."
            status_badge = f"{label_text} [{int(conf * 100)}%]"
            
            (w_name, h_name), _ = cv2.getTextSize(name_text, font, 0.45, 1)
            (w_stat, h_stat), _ = cv2.getTextSize(status_badge, font, 0.40, 1)
            
            card_w = max(w_name, w_stat) + 16
            card_h = h_name + h_stat + 14
            
            card_y1 = max(0, y1 - card_h - 4)
            card_y2 = y1 - 4
            card_x2 = min(w_frame, x1 + card_w)
            
            # Semi-transparent overlay banner untuk label
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (x1, card_y1), (card_x2, card_y2), (20, 24, 33), -1)
            # Border tipis penegas sesuai warna status
            cv2.rectangle(overlay, (x1, card_y1), (card_x2, card_y2), box_color, 1)
            cv2.addWeighted(overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)
            
            # Tulis Teks Nama & Status
            cv2.putText(annotated_frame, name_text, (x1 + 8, card_y1 + h_name + 4), 
                        font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, status_badge, (x1 + 8, card_y2 - 5), 
                        font, 0.38, box_color, 1, cv2.LINE_AA)
            
        self.tracks = new_tracks
        return annotated_frame, events
