@echo off
title Smart Face Attendance & Mask Compliance System (MFD Vision 60FPS)
echo ======================================================================
echo    🚀 SISTEM ABSENSI CERDAS & MASKED FACE DETECTION (MFD)
echo    Deep Learning: YOLOv8 MFD + FaceNet InceptionResnetV1 (CUDA)
echo    Engine: Flask Web REST API + Modern Responsive JS/Tailwind UI
echo ======================================================================
echo.
echo [1/2] Menyiapkan server lokal di http://127.0.0.1:5000 ...
echo [2/2] Browser akan otomatis terbuka dalam 3 detik...
echo.

:: Buka browser otomatis
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"

:: Jalankan server Python Flask
"C:\Users\Muhammad Lukman\AppData\Local\Programs\Python\Python310\python.exe" attendance_web\server.py

pause
