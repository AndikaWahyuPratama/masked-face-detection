# 😷 Masked Face Detection (MFD) using YOLOv8n

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Complete](https://img.shields.io/badge/Status-Completed%20%2830%20Epochs%29-brightgreen.svg)]()

> **Tugas Besar Mata Kuliah Deep Learning**  
> Sistem deteksi spasial dan klasifikasi kepatuhan penggunaan masker wajah secara *real-time* berbasis arsitektur **YOLOv8 Nano (YOLOv8n)** pada **Face Mask Label Dataset (FMLD)**.

---

## 🌟 Ringkasan Proyek

Proyek ini mengatasi masalah "kepatuhan semu" (*false compliance*) pada protokol pemakaian masker wajah di ruang publik. Bukan sekadar klasifikasi biner (*Mask / No Mask*), sistem ini melokalisasi posisi wajah (*bounding box*) sekaligus mendeteksi 3 kategori:

1. **`masked_face`** (Class 0): Memakai masker dengan benar (menutup hidung dan mulut).
2. **`unmasked_face`** (Class 1): Tidak memakai masker sama sekali.
3. **`incorrectly_masked_face`** (Class 2): Memakai masker keliru / melorot (di bawah hidung atau di dagu).

### 🏆 Hasil Kunci (*Key Achievements*)
- **Ukuran Model Super Ringkas:** Hanya **6.48 MB** (`paket_hasil/masked_face_yolov8n_best.pt`), sangat optimal untuk *edge devices* (CCTV cerdas, Raspberry Pi, laptop standar).
- **Performa Pengujian (FMLD Test Set - 7.152 Citra):**
  - **mAP@50 Keseluruhan:** **84.3%**
  - **`masked_face`:** **97.3%**
  - **`unmasked_face`:** **86.3%**
  - **`incorrectly_masked_face`:** **69.3%** *(Sangat solid untuk kelas langka ~2.5% data)*
- **F1-Score Optimal:** **0.81** pada *Confidence Threshold* **0.269**.
- **Real-Time Speed:** Mampu menjalankan inferensi video/webcam langsung tanpa *lag* (>30 FPS).

---

## 📊 Hasil Evaluasi & Metrik

| Kelas | Citra Uji (Test) | Bounding Box Ground Truth | mAP@0.5 | Keterangan |
| :--- | :---: | :---: | :---: | :--- |
| **All Classes** | **7.152** | **12.688** | **84.3%** | **Performa Menyeluruh Sangat Tinggi** |
| `masked_face` | - | 4.929 | **97.3%** | Hampir Sempurna (*Reliable*) |
| `unmasked_face` | - | 7.435 | **86.3%** | Akurat membedakan wajah terbuka |
| `incorrectly_masked_face` | - | 324 | **69.3%** | Berhasil mendeteksi masker melorot |

Grafik evaluasi lengkap (PR Curve, F1 Curve, Confusion Matrix) dapat dilihat pada folder [`runs/detect/test_e30b/`](runs/detect/test_e30b/) dan laporan lengkap [`reports/`](reports/).

---

## 📁 Struktur Direktori Repositori

```text
mfd/
├── paket_hasil/
│   └── masked_face_yolov8n_best.pt   # Bobot model terbaik hasil training (6.48 MB)
├── reports/
│   ├── class_distribution.png        # Grafik distribusi kelas per split
│   ├── dataset_check.txt             # Laporan validasi dataset (0 errors)
│   └── samples.png                   # Visualisasi sampel bounding box ground-truth
├── demo_in/                          # 12 sampel citra uji untuk demonstrasi cepat
├── runs/
│   └── detect/
│       ├── fmld_v8n_e30b/            # Hasil pelatihan 30 epoch (results.csv, curves)
│       ├── test_e30b/                # Hasil evaluasi test set resmi (PR, F1, Confusion Matrix)
│       └── demo_gambar/              # Hasil prediksi visual gambar demo
├── fmld_yolo/
│   ├── data.yaml                     # Konfigurasi kelas dan path YOLO
│   └── dataset_summary.json          # Statistik jumlah bounding box per subset
├── FMLD/                             # Skrip pendukung dari repositori FMLD resmi
│   ├── mask-test.py
│   ├── show_save_gt.py
│   └── README.md
├── prepare_fmld.py                   # Skrip konversi PASCAL VOC XML -> YOLO format
├── check_dataset.py                  # Skrip audit mutu dan verifikasi dataset
├── requirements.txt                  # Daftar dependensi Python
├── LAPORAN_PROYEK_MFD_END_TO_END.md  # Dokumentasi & laporan komprehensif akademik
└── README.md                         # Dokumentasi utama repositori
```

---

## 🚀 Panduan Memulai Cepat (Quickstart)

### 1. Instalasi Lingkungan
Pastikan Python 3.10+ sudah terpasang, lalu instal dependensi:
```bash
pip install -r requirements.txt
```
*(Opsional)* Jika memiliki GPU NVIDIA, instal PyTorch dengan akselerasi CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

### 2. Inferensi Menggunakan Bobot Terbaik
Model siap pakai sudah tersedia di folder `paket_hasil/`.

**A. Menguji Gambar Uji (Batch Demo):**
```bash
yolo detect predict model=paket_hasil/masked_face_yolov8n_best.pt source=demo_in conf=0.25 save=True
```
Hasil deteksi tersimpan di folder `runs/detect/predict/`.

**B. Menguji Melalui Kamera / Webcam Langsung (Real-Time):**
```bash
yolo detect predict model=paket_hasil/masked_face_yolov8n_best.pt source=0 conf=0.25 show=True
```
Tekan tombol `q` pada jendela kamera untuk keluar.

---

## 🛠️ Reproduksi Pipeline (Data & Training)

Jika Anda ingin mereproduksi seluruh siklus dari awal:

### 1. Persiapan Dataset (Data Engineering)
Unduh dataset gambar [MAFA](https://imsg.ac.cn/research/maskedface.html) dan [WIDER FACE](http://shuoyang1213.me/WIDERFACE), lalu ekstrak anotasi `FMLD_annotations.zip`:
```bash
python prepare_fmld.py \
  --ann_dir ann_unzip/FMLD_annotations \
  --images_root data/MAFA data/WIDER \
  --out fmld_yolo
```

### 2. Verifikasi Integritas Data
```bash
python check_dataset.py --data fmld_yolo --out reports
```

### 3. Pelatihan Model (30 Epochs)
```bash
yolo detect train \
  data=fmld_yolo/data.yaml \
  model=yolov8n.pt \
  imgsz=640 \
  epochs=30 \
  patience=10 \
  batch=16 \
  workers=4 \
  device=0 \
  name=fmld_v8n_e30b
```

### 4. Evaluasi Mandiri pada Test Set
```bash
yolo detect val \
  model=paket_hasil/masked_face_yolov8n_best.pt \
  data=fmld_yolo/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=test_eval
```

---

## 📖 Laporan Akademik Lengkap

Penjelasan mendalam mengenai teori, turunan matematis loss function, analisis arsitektur YOLOv8n, analisis confusion matrix, serta rekomendasi pembagian kerja tim dapat dibaca pada file:
👉 **[`LAPORAN_PROYEK_MFD_END_TO_END.md`](LAPORAN_PROYEK_MFD_END_TO_END.md)**

---

## 👥 Tim Pengembang (Kelompok Tugas Besar)
Proyek ini dikembangkan oleh tim beranggotakan 3 orang:
1. **Data Engineering & Quality Assurance** (`prepare_fmld.py`, `check_dataset.py`, Stratifikasi)
2. **Deep Learning & Model Training** (Arsitektur YOLOv8n, Hyperparameter Tuning, CUDA Optimization)
3. **Evaluation, Testing, & Deployment** (Analisis Metrik Test Set, Inferensi Gambar & Webcam Demo)

---

## 📜 Referensi & Sitasi
1. Batagelj, B., Peer, P., Štruc, V., & Dobrišek, S. (2021). *How to Correctly Detect Face-Masks for COVID-19 from Visual Information?* Applied Sciences, 11(5), 2070. [DOI: 10.3390/app11052070](https://doi.org/10.3390/app11052070)
2. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Version 8.0.0). [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
3. Ge, S., Li, J., Ye, Q., & Luo, Z. (2017). *Detecting Masked Faces in the Wild with LLE-CNNs*. CVPR 2017.
4. Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016). *WIDER FACE: A Face Detection Benchmark*. CVPR 2016.
