@echo off
setlocal
cd /d "%~dp0aerosplat_system"
if not exist static\vendor mkdir static\vendor
where npm >nul 2>nul
if errorlevel 1 (
  echo Node.js/npm is not installed or not on PATH.
  echo Install Node.js LTS, then rerun this script.
  exit /b 1
)
if not exist package.json npm init -y >nul
npm install @mkkellogg/gaussian-splats-3d@0.4.7 three@0.160.0
if errorlevel 1 exit /b 1
copy /Y node_modules\three\build\three.min.js static\vendor\three.min.js >nul
copy /Y node_modules\@mkkellogg\gaussian-splats-3d\build\gaussian-splats-3d.umd.cjs static\vendor\gaussian-splats-3d.umd.js >nul
if not exist static\vendor\three.min.js exit /b 1
if not exist static\vendor\gaussian-splats-3d.umd.js exit /b 1
echo.
echo Local Gaussian viewer installed successfully.
echo.
echo Files:
echo   static\vendor\three.min.js
echo   static\vendor\gaussian-splats-3d.umd.js
endlocal
