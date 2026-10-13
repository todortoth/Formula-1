# ==============================================================================
# F1 Parquet Dataset Scraper - Windows/PowerShell
# ==============================================================================

if (!(Test-Path "scraping_logs")) { New-Item -ItemType Directory -Path "scraping_logs" | Out-Null }

& ".\venv\Scripts\Activate.ps1"

2019..2025 | ForEach-Object {
    Year = _
    Start-Process powershell -ArgumentList "-NoProfile -Command & { .\scrape_data.ps1 -TargetYear \$Year }" `
        -RedirectStandardOutput "scraping_logs\output_\$Year.log" `
        -RedirectStandardError "scraping_logs\error_$Year.log" `
        -WindowStyle Hidden
}
