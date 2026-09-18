# Dzongkha LLM — one-shot env setup (PowerShell)
# Run from C:\Users\tharc\dzongkha-llm after files are in place.
$ErrorActionPreference = "Stop"
$Project = "C:\Users\tharc\dzongkha-llm"
$Py311 = "C:\Users\tharc\AppData\Local\Programs\Python\Python311\python.exe"

Set-Location $Project
if (-not (Test-Path $Py311)) { throw "Python 3.11 not found at $Py311" }

Write-Host "== Creating venv =="
& $Py311 -m venv .venv
& "$Project\.venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "== Installing torch (cu124) =="
try {
  & "$Project\.venv\Scripts\pip.exe" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
} catch {
  Write-Host "cu124 failed; trying cu121..."
  & "$Project\.venv\Scripts\pip.exe" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

Write-Host "== Installing requirements =="
& "$Project\.venv\Scripts\pip.exe" install -r requirements.txt

Write-Host "== check_env =="
& "$Project\.venv\Scripts\python.exe" scripts\check_env.py
