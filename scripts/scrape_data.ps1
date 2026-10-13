# ==============================================================================
# F1 Parquet Dataset Scraper - Windows/PowerShell
# ==============================================================================

param (
    [Parameter(Mandatory=\$true)]
    [int]\$TargetYear
)

\$BaseOutDir = "parquet_datasets"
if (!(Test-Path \$BaseOutDir)) {
    New-Item -ItemType Directory -Path \$BaseOutDir | Out-Null
}

Write-Host "Starting full-race F1 Parquet extraction for season \$TargetYear..." -ForegroundColor Cyan

for (\$round = 1; round -le 25; round++) {
    DatasetPath = Join-Path BaseOutDir "\({TargetYear}_round_\){round}.ds"

    if (Test-Path \$DatasetPath) {
        Write-Host "[Round \(round] dataset already exists at\)DatasetPath. Skipping." -ForegroundColor DarkGray
        continue
    }

    Write-Host "[Round \$round] Extracting FULL RACE dataset for TargetYear (Round round)..." -ForegroundColor Green

    python main.py `
        --year \$TargetYear `
        --event $round `
        --session "R" `
        --continuous `
        --from-lap 1 `
        --to-lap 99 `
        --hz 10 `
        --format parquet `
        --out-dir \$DatasetPath `
        --no-gif `
        --no-snapshots `
        --no-tables `
        --no-plots `
        --no-export

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[Round $round] Finished or $round does not exist for $TargetYear. Finalizing year." -ForegroundColor DarkYellow
        if (Test-Path $DatasetPath) {
            Remove-Item -Recurse -Force $DatasetPath -ErrorAction SilentlyContinue
        }
        break
    }
}

Write-Host "`nScraping complete for season \$TargetYear!" -ForegroundColor Cyan
