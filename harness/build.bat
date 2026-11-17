@echo off
setlocal
cd /d "%~dp0"

where docker >nul 2>nul
if errorlevel 1 (
  echo [ERROR] docker not found.
  echo         Install Docker Desktop, or use the GitHub Actions route in README.md
  pause
  exit /b 1
)

echo [1/3] docker build --platform linux/amd64 ...
docker build --platform linux/amd64 -t rsi-harness:v1 .
if errorlevel 1 goto fail

echo [2/3] describe + offline selftest A/B/C ...
docker run --rm rsi-harness:v1 describe
if errorlevel 1 goto fail
docker run --rm --user 0 --entrypoint python3 rsi-harness:v1 /opt/harness-selftest/selftest.py --task A
if errorlevel 1 goto fail
docker run --rm --user 0 --entrypoint python3 rsi-harness:v1 /opt/harness-selftest/selftest.py --task B
if errorlevel 1 goto fail
docker run --rm --user 0 --entrypoint python3 rsi-harness:v1 /opt/harness-selftest/selftest.py --task C
if errorlevel 1 goto fail

echo [3/3] docker save ...
docker save rsi-harness:v1 -o rsi-harness-v1.tar
if errorlevel 1 goto fail

echo.
echo OK  ->  %~dp0rsi-harness-v1.tar
dir rsi-harness-v1.tar
pause
exit /b 0

:fail
echo.
echo BUILD FAILED
pause
exit /b 1
