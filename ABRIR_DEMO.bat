@echo off
rem Doble clic: demo con fotos del conjunto test.
rem Arrastrar una foto o carpeta sobre este archivo: inspecciona esa foto o carpeta.
cd /d "%~dp0"
".venv\Scripts\python.exe" demo_fundivision.py %*
pause
