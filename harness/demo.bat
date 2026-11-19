@echo off
chcp 65001 >nul
setlocal

rem Offline demo: three training rounds + predict, using the built image (for screen recording)
rem   demo.bat            task A, 3 rounds, run straight through
rem   demo.bat A pause    task A, wait for Enter after each round (narrate while recording)
rem   demo.bat C          task C (engine RUL), 2 rounds, validation limited to 2000 rows

set IMAGE=rsi-harness:v3
set TASK=%~1
if "%TASK%"=="" set TASK=A
set PAUSE=%~2

set FOLDER=cpu-a
set VALROWS=0
set ROUNDS=3
if /i "%TASK%"=="B" set FOLDER=cpu-b
if /i "%TASK%"=="C" set FOLDER=cpu-c
if /i "%TASK%"=="C" set VALROWS=2000
if /i "%TASK%"=="C" set ROUNDS=2

set LABS=%~dp0..\labs

docker image inspect %IMAGE% >nul 2>&1
if errorlevel 1 (
  echo [x] image %IMAGE% not found, run build.bat first
  exit /b 1
)

set TTY=
set EXTRA=
if /i "%PAUSE%"=="pause" (
  set TTY=-it
  set EXTRA=--pause
)

docker run --rm --user 0 --entrypoint python3 %TTY% -v "%LABS%:/labs:ro" %IMAGE% ^
  /opt/harness-selftest/selftest.py --task %TASK% --rounds %ROUNDS% --demo ^
  --data-dir /labs/%FOLDER%/public --parent /labs/%FOLDER%/checkpoint.npz ^
  --train-rows 500 --val-rows %VALROWS% %EXTRA%

exit /b %errorlevel%
