@echo off
setlocal
cd /d "%~dp0aerosplat_system"
if not exist package.json (
  echo package.json not found.
  exit /b 1
)
if not exist node_modules (
  echo Installing browser viewer dependencies...
  call npm install
  if errorlevel 1 exit /b 1
)
if not exist static\vendor mkdir static\vendor
copy /Y node_modules\three\build\three.module.js static\vendor\three.module.js
copy /Y node_modules\@mkkellogg\gaussian-splats-3d\build\gaussian-splats-3d.module.js static\vendor\gaussian-splats-3d.module.js
if errorlevel 1 (
  echo Failed to copy viewer bundles.
  exit /b 1
)
echo.
echo Viewer bundles ready:
echo   static\vendor\three.module.js
echo   static\vendor\gaussian-splats-3d.module.js
exit /b 0
