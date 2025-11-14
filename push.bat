@echo off
cd /d %~dp0

git add .
git commit -m "Auto update %date% %time%"
git push origin main

echo.
echo ================================
echo  Git push 完成！
echo ================================
pause
