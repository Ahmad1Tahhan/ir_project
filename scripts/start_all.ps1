# Launch all SOA services + the Streamlit UI in separate windows.
# Usage:  ./scripts/start_all.ps1   (run from the project root)
#
# Stops cleanly by closing the spawned windows. Each service can also be started
# on its own with the uvicorn command shown in its module docstring.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"

function Start-Svc($name, $module, $port) {
    Write-Host "Starting $name on port $port ..."
    Start-Process -FilePath $py `
        -ArgumentList "-m", "uvicorn", "$module`:app", "--host", "127.0.0.1", "--port", "$port" `
        -WorkingDirectory $root
}

Start-Svc "retrieval"     "services.retrieval_service"   8001
Start-Svc "preprocessing" "services.preprocessing_service" 8002
Start-Svc "refinement"    "services.refinement_service"  8003
Start-Svc "evaluation"    "services.evaluation_service"  8004
# RAG service (loads a LOCAL LLM into GPU on first /chat call; model is cached offline):
Start-Svc "rag"           "services.rag_service"         8005

Start-Sleep -Seconds 4
Start-Svc "gateway"       "services.gateway"             8000

Start-Sleep -Seconds 3
Write-Host "Starting Streamlit UI ..."
Start-Process -FilePath $py -ArgumentList "-m", "streamlit", "run", "ui/app.py" -WorkingDirectory $root

Write-Host ""
Write-Host "All up. Gateway: http://127.0.0.1:8000/health   UI: http://localhost:8501"
