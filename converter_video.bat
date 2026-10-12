@echo off
rem Arraste um video (.mp4) para cima deste arquivo para gerar um .vmd por pessoa.
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
  echo Arraste um video para cima deste arquivo.
  pause
  exit /b 1
)
python video2vmd.py "%~1" --preview
pause
