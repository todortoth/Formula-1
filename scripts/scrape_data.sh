#!/usr/bin/env bash

# ==============================================================================
# F1 Parquet Dataset Bulk Scraper (Full Races: 2019 - 2025) - Linux/Bash
# ==============================================================================

BASE_OUT_DIR="parquet_datasets"
mkdir -p "$BASE_OUT_DIR"

START_YEAR=2019
END_YEAR=2025

echo -e "\033[36mStarting full-race F1 Parquet extraction for seasons $START_YEAR to $END_YEAR...\033[0m"

for ((year=START_YEAR; year<=END_YEAR; year++)); do
    echo -e "\n--------------------------------------------------"
    echo -e "\033[33mProcessing Season: $year\033[0m"
    echo -e "--------------------------------------------------"

    for ((round=1; round<=25; round++)); do

        DATASET_PATH="$BASE_OUT_DIR/${year}_round_${round}.ds"

        if [ -d "$DATASET_PATH" ]; then
            echo -e "\033[90m[Round $round] Dataset already exists at $DATASET_PATH. Skipping.\033[0m"
            continue
        fi

        echo -e "\033[32m[Round $round] Extracting FULL RACE dataset for $year (Round $round)...\033[0m"

        python main.py \
            --year "$year" \
            --event "$round" \
            --session "R" \
            --continuous \
            --from-lap 1 \
            --to-lap 99 \
            --hz 10 \
            --format parquet \
            --out-dir "$DATASET_PATH" \
            --no-gif \
            --no-snapshots \
            --no-tables \
            --no-plots \
            --no-export

        EXIT_CODE=$?

       if [ $EXIT_CODE -ne 0 ]; then
            echo -e "\033[33m[Round $round] Finished or round $round does not exist for $year. Moving to next year.\033[0m"

            if [ -d "$DATASET_PATH" ]; then
                rm -rf "$DATASET_PATH"
            fi

            break
        fi
    done
done

echo -e "\n\033[36mBulk scraping complete! All available full-race parquet datasets are stored under '$BASE_OUT_DIR/'.\033[0m"