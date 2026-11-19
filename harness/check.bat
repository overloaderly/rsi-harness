@echo off
rem Offline self-check for screen recording: describe + protocol + predict.
rem   check.bat
setlocal
set IMAGE=rsi-harness:v3
pushd "%~dp0"
set HERE=%CD%
popd
pushd "%~dp0..\labs"
set LABS=%CD%
popd

docker image inspect %IMAGE% >nul 2>&1
if errorlevel 1 (
  echo [x] image %IMAGE% not found, run build.bat first
  exit /b 1
)

echo === 1/4 describe
docker run --rm %IMAGE% describe
echo.
echo === 2/4 protocol self-check
docker run --rm --user 0 --entrypoint python3 -v "%HERE%:/work:ro" -v "%LABS%:/labs:ro" %IMAGE% /work/tools/probe_protocol.py
echo.
echo === 3/4 predict check, task A, 50 rows
docker run --rm --user 0 --entrypoint python3 -v "%HERE%:/work:ro" -v "%LABS%:/labs:ro" %IMAGE% /work/tools/verify_predict.py /labs/cpu-a/checkpoint.npz /labs/cpu-a/public/validation.jsonl A 50
echo.
echo === 4/4 predict check, task C, 50 rows
docker run --rm --user 0 --entrypoint python3 -v "%HERE%:/work:ro" -v "%LABS%:/labs:ro" %IMAGE% /work/tools/verify_predict.py /labs/cpu-c/checkpoint.npz /labs/cpu-c/public/validation.jsonl C 50
echo.
exit /b %errorlevel%
