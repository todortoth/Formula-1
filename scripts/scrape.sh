# ==============================================================================
# F1 Parquet Dataset Scraper
# ==============================================================================

mkdir -p scraping_logs

for year in {2019..2025}; do
    nohup bash -c "source /home/todortoth/F1/bin/activate && scripts/scrape_data.sh $year" > "scraping_logs/output_${year}.log" 2>&1 &
done

echo "All F1 scraping background workers initiated for 2019-2025!"
echo "Check progress using: tail -f scraping_logs/output_2023.log"
