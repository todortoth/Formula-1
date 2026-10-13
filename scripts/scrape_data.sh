#!/usr/bin/env bash

# ==============================================================================
# F1 Parquet Dataset Scraper (Single Year Worker)
# ==============================================================================

if [ -z "$1" ]; then
    echo "Error: No year specified. Usage: $0 <year>"
    exit 1
fi

TARGET_YEAR=$1
BASE_OUT_DIR="parquet_datasets"
mkdir -p "$BASE_OUT_DIR"

echo -e "\033[36mStarting full-race F1 Parquet extraction for season $TARGET_YEAR...\033[0m"

for ((round=1; round<=25; round++)); do
    # Pad the round number with a leading zero (e.g., 1 becomes 01)
    printf -v padded_round "%02d" "$round"

    DATASET_PATH="$BASE_OUT_DIR/${TARGET_YEAR}_round_${padded_round}.ds"

    if [ -d "$DATASET_PATH" ]; then
        echo -e "[Round $padded_round] Dataset already exists at $DATASET_PATH. Skipping."
        continue
    fi

    echo -e "[Round $padded_round] Extracting FULL RACE dataset for $TARGET_YEAR (Round $padded_round)..."

    python main.py \
        --year "$TARGET_YEAR" \
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
        echo "[WARNING] Round $padded_round failed with exit code $EXIT_CODE."

        if [ -d "$DATASET_PATH" ]; then
            rm -rf "$DATASET_PATH"
        fi

        echo "[Round $padded_round] Encountered data error or invalid round. Concluding loop for $TARGET_YEAR."
        break
    fi
done

echo -e "\n\033[32mScraping complete for season $TARGET_YEAR!\033[0m"
exit 0
