@echo off
REM DataOrbit - run the whole pipeline end to end with one command (Windows).
REM Same as run_all.sh: setup if needed, then bulk - fresh - stars - export - verify.
REM Stops at the first failure. Needs only .env with your keys in this folder;
REM it creates the venv and installs everything itself when missing.
REM Run from PowerShell or cmd, or double-click this file.
setlocal
cd /d %~dp0
if not exist .env (
  echo No .env found. Copy .env.example to .env and add your keys, then run this again.
  exit /b 1
)
if not exist .venv (
  echo No venv found - creating it and installing everything, one-time setup...
  python -m venv .venv || exit /b 1
  call .venv\Scripts\activate.bat
  pip install -e ".[dev,playwright,llm]" || exit /b 1
  echo Installing the browser for Hirist - optional, continues if this fails...
  playwright install chromium
) else (
  call .venv\Scripts\activate.bat
  python -c "import pydantic, aiohttp, gspread" 2>nul
  if errorlevel 1 (
    echo Venv found but packages missing - installing...
    pip install -e ".[dev,playwright,llm]" || exit /b 1
    playwright install chromium
  )
)
echo === 1/5 bulk run — about an hour for 1,000 of each ===
python scripts\run_bulk.py --no-db || exit /b 1
echo === 2/5 fresh jobs + news (last 24h) ===
python scripts\run_fresh.py --no-db || exit /b 1
echo === 3/5 GitHub stars ===
python scripts\enrich_stars.py || exit /b 1
echo === 4/5 export to Google Sheet ===
python scripts\export_batched.py || exit /b 1
echo === 5/5 verify ===
python scripts\verify_sheets.py || exit /b 1
echo DONE — open the Sheet and check all 6 tabs.
