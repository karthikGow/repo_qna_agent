# Demo script for Repo Q&A Agent (assumes API is already running on :8000)
# Usage:  .\demo.ps1
# Notes: Adjust $Repo if you want to target a different repository.

param(
  [string]$Repo = "karthikGow/repo_qna_agent",
  [string]$ApiUrl = "http://127.0.0.1:8000/chat"
)

Write-Host "Demo against repo: $Repo" -ForegroundColor Cyan

# Optional: ensure venv Python is used if running inside an activated shell
$questions = @(
  "What was implemented in the last commit and by whom?",
  "When was the last deployment?",
  "When did we fix the favicon bug?",
  "When did we refactor the homepage?"
)

foreach ($q in $questions) {
  Write-Host "\n> $q" -ForegroundColor Yellow
  python cli.py --repo $Repo --api-url $ApiUrl -- "$q"
}

Write-Host "\nDone." -ForegroundColor Green
