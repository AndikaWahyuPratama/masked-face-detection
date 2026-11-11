@echo off
title Smart Face Attendance & Mask Compliance System (MFD Vision 60FPS)
cd /d "%~dp0"
echo ======================================================================
echo    🚀 SISTEM ABSENSI CERDAS & MASKED FACE DETECTION (MFD)
echo    Deep Learning: YOLOv8 MFD + FaceNet InceptionResnetV1 (CUDA)
echo    Engine: Flask Web REST API + Modern Responsive JS/Tailwind UI
echo ======================================================================
echo.
echo [*] Sedang memuat model AI ke GPU CUDA (membutuhkan waktu ~10 detik)...
echo [*] Browser kamu akan otomatis terbuka setelah server siap!
echo.

"C:\Users\Muhammad Lukman\AppData\Local\Programs\Python\Python310\python.exe" -u attendance_web\server.py

echo.
echo Server telah dimatikan.
pause
