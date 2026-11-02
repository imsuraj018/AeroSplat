@echo off
setlocal
set "CONDA_ROOT=C:\Users\Admin\anaconda3"
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
call conda activate myenv
cd /d "%~dp0aerosplat_system"
python server.py
