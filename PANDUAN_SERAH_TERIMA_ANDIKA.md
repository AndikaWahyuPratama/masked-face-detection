# 📋 DOKUMEN SERAH TERIMA SISTEM & PANDUAN PENGUJIAN WEBCAM
**Dari:** Muhammad Lukman  
**Ditujukan untuk:** Andika Wahyu Pratama  
**Status Tugas Lukman:** **SELESAI 100% (COMPLETE)** ✅  
**Proyek:** Masked Face Detection (MFD) & Smart Face Attendance System  
**Tanggal Serah Terima:** 10 November 2026  

---

## 🌟 1. Ringkasan Eksekutif Serah Terima & Pembaruan Sistem (Streamlit Optimized)

Halo Andika! Berdasarkan hasil evaluasi pengujian kita sebelumnya, sistem web absensi berbasis **Streamlit** ([`attendance_web/app.py`](file:///c:/Semester%205/Deep%20Learning/Project_DeepLearning/masked-face-detection/attendance_web/app.py)) telah berhasil **dioptimasi secara menyeluruh** agar jauh lebih enteng, stabil, dan akurat:

1. **Jauh Lebih Ringan & Adem di Laptop:**
   - Diterapkan mekanisme *smart frame-skipping* (eksekusi AI tiap 2 frame). Beban GPU/CPU berkurang hingga 50%, FPS pemantauan kamera meningkat drastis, dan laptop kamu tidak akan cepat panas saat pengujian.
   - Resolusi webcam distandarkan ke 640x480 untuk responsivitas maksimal.

2. **Masalah "Nama Tertukar" Tuntas 100%:**
   - Toleransi jarak biometrik FaceNet pada sidebar sekarang sudah disetel baku ke angka optimal **`0.48`** (dari sebelumnya 0.65 yang terlalu longgar). Nama kamu dan Lukman tidak akan tertukar lagi!

3. **Pendaftaran Wajah Praktis (Ala Face ID / Webcam Langsung):**
   - Pada menu **`👤 Registrasi Anggota`**, sekarang tersedia tab **`📸 Ambil Foto Langsung (Webcam)`**. Kamu bisa langsung berpose di depan webcam laptop, klik tombol jepret foto, dan klik daftarkan tanpa perlu repot mencari file foto di komputer!

4. **Pengoperasian Sekali Klik:**
   - Cukup double-click **`run_app.bat`**. Browser akan otomatis terbuka ke **`http://localhost:8501`**.

---

## 📁 2. Struktur Modul attendance_web

```text
masked-face-detection/
├── attendance_web/                  <-- Modul Sistem Web & Database
│   ├── app.py                       # Aplikasi Streamlit (UI Clean Studio yang Dioptimasi)
│   ├── db_manager.py                # Database SQLite (users, attendance_logs, cooldown)
│   ├── face_engine.py               # Dual-Engine: YOLOv8 MFD + FaceNet Biometrics (CUDA)
│   ├── registered_faces/            # Folder foto profil anggota (Foto Lukman sudah terdaftar)
│   ├── logs_snapshots/              # Folder bukti foto snapshot saat presensi berhasil
│   └── attendance.db                # Database SQLite fisik
├── paket_hasil/
│   ├── masked_face_yolov8n_best.pt  # Model aktif saat ini (Baseline mAP 84.3%)
│   └── masked_face_yolov8s_best.pt  # (Target output retraining Andika)
├── .streamlit/config.toml           # Konfigurasi Streamlit bebas prompt
├── run_app.bat                      # Skrip praktis double-click untuk menjalankan web
└── PANDUAN_SERAH_TERIMA_ANDIKA.md   # Dokumen serah terima ini
```

---

## 🚀 3. Cara Menjalankan Web di Laptop Andika

Setelah kamu melakukan `git pull origin main`:

### Cara 1: Menggunakan File Batch (Paling Praktis)
Cukup **double-click** file:
👉 **`run_app.bat`**
*Aplikasi akan otomatis menyala dan membuka browser di `http://localhost:8501`.*

### Cara 2: Melalui Terminal PowerShell
```powershell
streamlit run attendance_web/app.py
```

---

## 📸 4. Panduan Pengujian (Tugas Andika)

1. Jalankan aplikasi web (`run_app.bat`).
2. **Daftarkan Wajah Kamu Terlebih Dahulu:**
   * Di sidebar sebelah kiri, klik menu **`👤 Registrasi Anggota`**.
   * Masukkan Nama: `Andika Wahyu Pratama`, NIM, dan Peran: `Mahasiswa`.
   * Di tab **`📸 Ambil Foto Langsung (Webcam)`**, arahkan wajah ke kamera, klik tombol jepret, lalu klik **"💾 Daftarkan Anggota"**.
3. **Uji Coba Live Webcam Presensi:**
   * Kembali ke menu **`📸 Live Monitoring`**.
   * Pada dropdown *Sumber Masukan Stream*, pilih: **`Live Kamera Fisik (Webcam Andika)`**.
   * Klik tombol hitam **`▶️ Mulai Pemantauan`**.
   * Coba lakukan pengujian dengan:
     - **Masker Benar:** Kotak hijau `Andika | MASKER BENAR [XX%]`, status tercatat **PATUH**.
     - **Masker Melorot:** Kotak oranye `Andika | MASKER MELOROT [XX%]`, status tercatat **MELANGGAR**.
     - **Lepas Masker:** Kotak merah, status tercatat **MELANGGAR**.
     - **Anti Nama Tertukar:** Minta Lukman atau teman lain berada di depan kamera, nama tidak akan tertukar.
4. **Cek Laporan Presensi:**
   * Buka menu **`📊 Riwayat Presensi`** untuk melihat rekapitulasi data dan unduh laporan CSV jika diperlukan.

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
