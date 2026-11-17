@echo off
REM Opens the MLflow UI on this project's experiment store, whatever folder you start it from.
REM Usage (CMD): double-click this file, or run  mlflow_ui.bat  [port]
REM (An absolute C:\ path cannot be used here: MLflow would read "C:" as a URI scheme.)
cd /d "%~dp0"
set PORT=%1
if "%PORT%"=="" set PORT=5000
echo Starting MLflow UI on http://127.0.0.1:%PORT%  (store: %CD%\mlruns)
echo Open the "chest-xray-pneumonia" experiment in the left sidebar. Press Ctrl+C to stop.
".venv\Scripts\mlflow.exe" ui --backend-store-uri ./mlruns --port %PORT%
