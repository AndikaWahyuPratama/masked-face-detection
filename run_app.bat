@echo off
setlocal
cd /d "%~dp0"
echo ======================================================================
echo    SMART FACE ATTENDANCE ^& MASK COMPLIANCE SYSTEM
echo    Deep Learning: YOLOv8 MFD + FaceNet (Streamlit Engine)
echo ======================================================================
echo.
echo Sedang membuka aplikasi Streamlit...
echo Web browser akan otomatis terbuka dalam beberapa detik.
echo.
echo. | "C:\Users\Muhammad Lukman\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run attendance_web/app.py --browser.gatherUsageStats=false
pause
