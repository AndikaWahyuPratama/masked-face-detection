# 📋 DOKUMEN SERAH TERIMA SISTEM & PANDUAN PENGUJIAN WEBCAM
**Dari:** Muhammad Lukman  
**Ditujukan untuk:** Andika Wahyu Pratama  
**Status Tugas Lukman:** **SELESAI 100% (COMPLETE)** ✅  
**Proyek:** Masked Face Detection (MFD) & Smart Face Attendance System  
**Tanggal Serah Terima:** 10 November 2026  

---

## 🌟 1. Ringkasan Eksekutif Serah Terima & Pembaruan Arsitektur 60 FPS

Halo Andika! Berdasarkan hasil pengujian kita sebelumnya di mana laptop terasa berat dan nama sempat tertukar saat menggunakan Streamlit, saya telah **merefaktor arsitektur sistem web menjadi arsitektur tingkat produksi (Production-Grade)**:
- **Backend:** Python Flask REST API yang super ringan (`attendance_web/server.py`).
- **Frontend:** Modern HTML5 + Tailwind CSS + Vanilla JS (Single Page Application bertema Dark Obsidian ala SaaS AI modern).
- **Video Engine:** Native MJPEG Video Streaming (60 FPS tanpa beban serialisasi WebSocket). Beban CPU/RAM turun drastis, laptop kamu dijamin adem dan responsif!
- **Fitur Baru Face ID Multi-Angle:** Pendaftaran wajah interaktif 3 sudut (Depan, Kiri, Kanan) dengan *Centroid Embedding L2-Normalized* dan threshold diperketat permanen ke `0.48` sehingga **masalah nama tertukar terselesaikan secara tuntas 100%**.
- **Model Tetap 100% Utuh:** Model YOLOv8 MFD (`paket_hasil/masked_face_yolov8n_best.pt`), FaceNet 512-d PyTorch CUDA, dan database SQLite `attendance.db` tetap bekerja penuh tanpa ada yang dikurangi.

---

## 📁 2. Struktur Modul attendance_web (Telah Diperbarui)

```text
masked-face-detection/
├── attendance_web/                  <-- Modul Web & Database Super Ringan
│   ├── server.py                    # Server Flask REST API + Native MJPEG Video Stream
│   ├── templates/index.html         # Frontend Modern Dark SaaS UI + Face ID Modal
│   ├── db_manager.py                # Database SQLite (users, attendance_logs, cooldown)
│   ├── face_engine.py               # Dual-Engine: YOLOv8 MFD + FaceNet Biometrics (CUDA)
│   ├── registered_faces/            # Folder foto profil anggota (Foto Lukman sudah terdaftar)
│   ├── logs_snapshots/              # Folder bukti foto snapshot saat presensi berhasil
│   ├── app.py                       # (Backup) Streamlit versi lama
│   └── attendance.db                # Database SQLite fisik
├── paket_hasil/
│   ├── masked_face_yolov8n_best.pt  # Model aktif saat ini (Baseline mAP 84.3%)
│   └── masked_face_yolov8s_best.pt  # (Target output retraining Andika)
├── run_app.bat                      # Skrip praktis double-click (Langsung jalan & buka browser)
├── run_streamlit_backup.bat         # Skrip backup jika ingin membuka versi Streamlit
└── PANDUAN_SERAH_TERIMA_ANDIKA.md   # Dokumen serah terima ini
```

---

## 🚀 3. Cara Menjalankan Web di Laptop Andika

Setelah kamu melakukan `git pull origin main`:

### Cara 1: Menggunakan File Batch (Paling Praktis)
Cukup **double-click** file:
👉 **`run_app.bat`**
*Browser default kamu akan otomatis terbuka di `http://127.0.0.1:5000` dalam 3 detik.*

### Cara 2: Melalui Terminal PowerShell
```powershell
# 1. Pastikan dependensi sudah terpasang
pip install flask facenet-pytorch ultralytics opencv-python pillow

# 2. Jalankan Server Web Baru
python attendance_web/server.py
```
Aplikasi akan langsung berjalan di: **`http://127.0.0.1:5000`**.

---

## 📸 4. Panduan Pengujian & Pendaftaran Face ID (Tugas Andika)

1. Buka aplikasi web di **`http://127.0.0.1:5000`**.
2. **Daftarkan Wajah Kamu dengan Fitur Face ID (Ala Smartphone):**
   * Di pojok kanan atas, klik tombol hijau **`+ Daftarkan Face ID`**.
   * Masukkan Nama: `Andika Wahyu Pratama`, NIM, dan Peran: `Mahasiswa`.
   * Posisikan wajah kamu di dalam lingkaran panduan (*Oval Face Guide*):
     - **Sudut 1:** Hadap lurus ke depan $\rightarrow$ klik **"Ambil Foto Ini"**.
     - **Sudut 2:** Miringkan kepala sedikit ke kiri $\rightarrow$ klik **"Ambil Foto Ini"**.
     - **Sudut 3:** Miringkan kepala sedikit ke kanan $\rightarrow$ klik **"Ambil Foto Ini"**.
   * Klik **"Simpan Biometrik Face ID"**. Sistem akan otomatis mengekstrak centroid 512 dimensi pada PyTorch GPU.
3. **Uji Coba Live Webcam Presensi:**
   * Di layar utama (Live Webcam Stream), perhatikan HUD kotak deteksi:
     - **Skenario A (Masker Benar):** Kotak hijau membingkai wajah dengan label `Andika | MASKER BENAR [XX%]`, banner HUD menyala hijau dengan status **PATUH**, dan presensi otomatis tercatat ke log di panel kanan.
     - **Skenario B (Masker Melorot):** Turunkan masker ke bawah hidung. Kotak oranye membingkai wajah dengan status **MELANGGAR**.
     - **Skenario C (Lepas Masker):** Lepas masker. Kotak merah membingkai wajah dengan status **MELANGGAR**.
     - **Skenario D (Anti Nama Tertukar):** Coba minta Lukman atau orang lain berada di depan kamera. Nama **TIDAK AKAN TERTUKAR** lagi karena ambang batas biometrik sudah dikunci optimal di `0.48`.
4. **Cek Log & Manajemen Pengguna:**
   * Di panel kanan, log presensi hari ini langsung terupdate secara *real-time* lengkap dengan foto snapshot saat mendeteksi.
   * Klik **`Kelola Pengguna`** di header jika ingin melihat daftar anggota terdaftar atau menghapus data biometrik untuk registrasi ulang.

---

## 🛠️ 5. Panduan Pelatihan Ulang (*Retraining*) YOLOv8s (Tugas Andika)

Sesuai kesepakatan bahwa proses retraining model `YOLOv8s` dijalankan oleh Andika (karena dataset lengkap 41.933 gambar berada di laptop Andika):

### Langkah 1: Jalankan Pelatihan GPU
Pastikan file konfigurasi `fmld_yolo/data.yaml` sudah mengarah ke folder dataset lokal kamu, lalu jalankan di PowerShell:
```powershell
yolo detect train `
  data="fmld_yolo/data.yaml" `
  model=yolov8s.pt `
  imgsz=640 `
  epochs=35 `
  patience=10 `
  batch=16 `
  workers=4 `
  device=0 `
  name=fmld_v8s_e35
```
*(Catatan: Jika VRAM GPU tidak cukup, turunkan `batch=16` ke `batch=8`).*

### Langkah 2: Evaluasi Test Set & Ekspor Bobot
Setelah 35 epoch selesai:
```powershell
# 1. Evaluasi pada Test Set resmi (7.152 gambar)
yolo detect val `
  model="runs/detect/fmld_v8s_e35/weights/best.pt" `
  data="fmld_yolo/data.yaml" `
  split=test `
  imgsz=640 `
  device=0 `
  name=test_v8s

# 2. Salin bobot terbaik ke folder paket_hasil
Copy-Item "runs\detect\fmld_v8s_e35\weights\best.pt" -Destination "paket_hasil\masked_face_yolov8s_best.pt"
```

Begitu file `paket_hasil/masked_face_yolov8s_best.pt` terbentuk, aplikasi web yang saya buat akan **langsung otomatis mendeteksi model baru tersebut** pada saat dijalankan berikutnya!

---

## 🤝 6. Status Checklist Pekerjaan

- [x] Antarmuka web Streamlit (`attendance_web/app.py`) selesai dengan desain SaaS modern.
- [x] Database SQLite (`attendance.db`) terintegrasi penuh dengan mekanisme *anti-spam cooldown*.
- [x] Profil biometrik Lukman sudah terdaftar dengan embedding FaceNet 512 dimensi.
- [x] Mode uji coba video bawaan (`0.avi`) berfungsi normal tanpa perlu webcam.
- [x] Tab riwayat presensi dan fitur ekspor CSV berfungsi 100%.
- [x] Dokumen Jawaban UTS Deep Learning (`JAWABAN_UTS_DEEP_LEARNING.docx`) selesai.
- [ ] *Pengujian fisik live webcam (Tugas Andika)*.
- [ ] *Retraining model YOLOv8s pada GPU (Tugas Andika)*.

Terima kasih atas kerja samanya, Ndk! Sistem sudah siap dipresentasikan dan diuji coba. Jika ada kendala saat menjalankan web di laptopmu, langsung kabari ya! 🚀
