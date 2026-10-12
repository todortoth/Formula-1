# ==============================================================================
# F1 Parquet Dataset Bulk Scraper (Full Races: 2019 - 2025) - Windows/PowerShell
# ==============================================================================

$BaseOutDir = "parquet_datasets"
if (!(Test-Path $BaseOutDir)) {
    New-Item -ItemType Directory -Path $BaseOutDir | Out-Null
}

$StartYear = 2019
$EndYear = 2025

Write-Host "Starting full-race F1 Parquet extraction for seasons $StartYear to$EndYear..." -ForegroundColor Cyan

for ($year = $StartYear; $year -le $EndYear; $year++) {
    Write-Host "`n--------------------------------------------------"
    Write-Host "Processing Season: $year" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------"

    for($round = 1; $round -le 25; $round++) {
        $DatasetPath = Join-Path $BaseOutDir "${year}_round_${round}.ds"

        if (Test-Path $DatasetPath)
        {
            Write-Host "[Round $round] dataset already exists at $DatasetPath. Skipping." -ForegroundColor DarkGray
            continue
        }

        Write-Host "[Round $round] Extracting FULL RACE dataset for $year (Round $round)..." -ForegroundColor Green

        python main.py `
            --year $year `
            --event $round `
            --session "R" `
            --continuous `
            --from-lap 1 `
            --to-lap 99 `
            --hz 10 `
            --format parquet `
            --out-dir $DatasetPath `
            --no-gif `
            --no-snapshots `
            --no-tables `
            --no-plots `
            --no-export

        if ($LASTEXITCODE -ne 0) {
            Write-Host "[Round $round] Finished or $round does not exist for $year. Moving to next year." -ForegroundColor DarkYellow
            if (Test-Path $DatasetPath) {
                Remove-Item -Recurse -Force $DatasetPath -ErrorAction SilentlyContinue
            }
            break
        }
    }
}

Write-Host "`nBulk scraping complete! All available full-race parquet datasets are stored under '$BaseOutDir/'." -ForegroundColor Cyan