# 📋 DOKUMEN SERAH TERIMA SISTEM & PANDUAN PENGUJIAN WEBCAM
**Dari:** Muhammad Lukman  
**Ditujukan untuk:** Andika Wahyu Pratama  
**Status Tugas Lukman:** **SELESAI 100% (COMPLETE)** ✅  
**Proyek:** Masked Face Detection (MFD) & Smart Face Attendance System  
**Tanggal Serah Terima:** 10 November 2026  

---

## 🌟 1. Ringkasan Eksekutif Serah Terima

Halo Andika! Seluruh integrasi **Sistem Web Absensi Cerdas & Kepatuhan Protokol Masker** (`attendance_web/`) beserta **Arsitektur Database SQLite** dan **Pipeline AI Biometrik (YOLOv8 + FaceNet)** telah berhasil saya bangun, uji, dan selesaikan secara tuntas.

Sistem ini telah dirancang dengan arsitektur **Clean SaaS Studio** (berdasarkan acuan desain modern) dan sudah teruji menghubungkan deteksi masker spasial ke database presensi dengan mekanisme *anti-duplikasi (cooldown 5 menit)*.

> 💡 **Fitur Otomatisasi Model (`face_engine.py`):**  
> Sistem sudah diprogram cerdas: saat ini sistem aktif menggunakan bobot `paket_hasil/masked_face_yolov8n_best.pt`. Begitu kamu selesai me-*retrain* model **YOLOv8s** dan meletakkannya di `paket_hasil/masked_face_yolov8s_best.pt`, aplikasi web akan **otomatis mendeteksi dan beralih ke YOLOv8s tanpa perlu mengubah kode satu baris pun!**

---

## 📁 2. Struktur Modul yang Dikerjakan Lukman

```text
masked-face-detection/
├── attendance_web/                  <-- Modul Sistem Web & Database (Selesai)
│   ├── app.py                       # Antarmuka Dashboard Streamlit (Clean Light Studio)
│   ├── db_manager.py                # Manajemen SQLite (users, attendance_logs, cooldown)
│   ├── face_engine.py               # Dual-Engine: YOLOv8 MFD + FaceNet Biometrics (CUDA)
│   ├── registered_faces/            # Folder foto profil anggota (Foto Lukman sudah terdaftar)
│   ├── logs_snapshots/              # Folder bukti foto snapshot saat presensi berhasil
│   ├── check_db.py                  # Skrip utilitas CLI untuk cek isi database
│   └── attendance.db                # Database SQLite fisik
├── paket_hasil/
│   ├── masked_face_yolov8n_best.pt  # Model aktif saat ini (Baseline mAP 84.3%)
│   └── masked_face_yolov8s_best.pt  # (Target output retraining Andika)
├── runs/detect/demo_webcam/0.avi    # Berkas video demo untuk pengujian mandiri
├── JAWABAN_UTS_DEEP_LEARNING.docx   # Dokumen Word jawaban UTS lengkap (No. 1-5 + Gambar)
├── run_app.bat                      # Skrip praktis double-click untuk menjalankan web
└── PANDUAN_SERAH_TERIMA_ANDIKA.md   # Dokumen serah terima ini
```

---

## 🚀 3. Panduan Cepat Menjalankan Web di Laptop Andika

Setelah kamu melakukan `git pull origin main`:

### Cara 1: Menggunakan File Batch (Paling Praktis)
Cukup **double-click** file:
👉 **`run_app.bat`**

### Cara 2: Melalui Terminal PowerShell
```powershell
# 1. Pastikan dependensi sudah terpasang
pip install streamlit facenet-pytorch ultralytics pandas pillow opencv-python

# 2. Jalankan Streamlit
streamlit run attendance_web/app.py
```
Aplikasi akan langsung terbuka di browser pada alamat: **`http://localhost:8501`**.

---

## 📸 4. Panduan Pengujian Live Kamera Fisik (Tugas Andika)

Karena di laptop saya kamera fisiknya sedang nonaktif, kamu bertugas menguji fungsi kamera langsung:

1. Buka aplikasi web di **`http://localhost:8501`**.
2. **Daftarkan Wajah Kamu Terlebih Dahulu:**
   * Di sidebar sebelah kiri, klik menu **`👤 Registrasi Anggota`**.
   * Masukkan Nama: `Andika Wahyu Pratama`, NIM, dan Peran: `Mahasiswa`.
   * Unggah pasfoto wajah kamu yang jelas, lalu klik **"💾 Daftarkan Anggota"**.
   * *Verifikasi:* Wajah kamu akan terpotong otomatis dan muncul di direktori anggota sebelah kanan.
3. **Uji Coba Live Webcam:**
   * Kembali ke menu **`📸 Live Monitoring`**.
   * Pada dropdown *Sumber Masukan Stream*, pilih: **`Live Kamera Fisik (Webcam Andika)`**.
   * Klik tombol hitam **`▶️ Mulai Pemantauan`**.
   * Webcam laptop kamu akan menyala. Coba lakukan 3 skenario ini di depan kamera:
     * **Skenario A (Masker Benar):** Kotak hijau akan membingkai wajah dengan label `Andika | MASKER BENAR [XX%]`, dan presensi tercatat dengan status **PATUH**.
     * **Skenario B (Masker Melorot):** Turunkan masker ke bawah hidung. Kotak oranye akan muncul dengan label `Andika | MASKER MELOROT [XX%]`, dan status tercatat **MELANGGAR**.
     * **Skenario C (Lepas Masker):** Lepas masker sepenuhnya. Kotak merah akan muncul dengan status **MELANGGAR**.
4. **Cek Laporan Presensi:**
   * Buka menu **`📊 Riwayat Presensi`**.
   * Periksa apakah nama kamu, jam presensi, status kepatuhan, dan persentase keyakinan AI sudah masuk ke tabel.
   * Uji tombol **`📥 Unduh Laporan Resmi (CSV / Excel)`** untuk mengunduh rekap kehadiran.

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
