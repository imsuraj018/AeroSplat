@echo off
setlocal
set "CONDA_ROOT=C:\Users\Admin\anaconda3"
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
call conda activate laya_env
set LAYA_HOST=127.0.0.1
set LAYA_PORT=8100
set LAYA_DEVICE=cpu
set LAYA_PRELOAD=0
set LAYA_LOG_LEVEL=info
laya-serve
