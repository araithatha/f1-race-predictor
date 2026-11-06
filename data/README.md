# Input data

These CSVs are a snapshot of the tables this model needs, produced by
**[f1-data-pipeline](https://github.com/araithatha/f1-data-pipeline)**. That pipeline pulls
Jolpica-F1, OpenF1 and Open-Meteo, cleans the data, joins it and runs 32 data-quality checks.
`_manifest.json` records when the snapshot was built and the row counts.

| File | What it holds |
|---|---|
| `dim_race.csv` | one row per Grand Prix, 2014 to Nov 2026, with race-day weather |
| `dim_driver.csv` | driver names and details |
| `fact_result.csv` | race results: grid, finish, status, points |
| `fact_qualifying.csv` | Q1/Q2/Q3 times and gap to pole |
| `fact_pit_stop.csv` | every pit stop since 2014 |
| `fact_practice.csv` | practice-session pace per driver, 2023+ |

## Using newer data

```bash
git clone https://github.com/araithatha/f1-data-pipeline
cd f1-data-pipeline && pip install -r requirements.txt
PYTHONPATH=src python -m f1pipeline.run --refresh

# then, back in this repo, point the model at the fresh output:
F1_DATA_DIR=../f1-data-pipeline/data/processed PYTHONPATH=src python -m f1model.run
```
