# Make Believe Football – Data Cleaner

Python utilities to clean and consolidate player stats from the NFL-Data repository.

## What this produces
- 2015–2020: season totals only (week = 0 per player/season)
- 2021–2024: weekly breakdowns (weeks 1–17) per player with NA where a player didn’t play
- Excludes defensive positions `DB`, `DL`, `LB`
- Output files written to `data/processed/`:
  - `nfl_weekly.csv`
  - `nfl_weekly.parquet` (requires `pyarrow`)

## Requirements
- Python 3.10+
- pandas, pyarrow (see `requirements.txt`)

## Setup
```bash
# 1) Clone this repo (after you push it to GitHub)
# git clone <your-repo-url>
# cd make-believe-football

# 2) Create a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Get the data source (clone locally)
git clone https://github.com/hvpkod/NFL-Data ~/Downloads/NFL-Data
```

## Run the cleaner
```bash
# from project root
python -m src.data.clean_nfl_data \
  --source-dir ~/Downloads/NFL-Data \
  --output-dir data/processed
```

You should see outputs under `data/processed/`.

## Quick verification
```python
import pandas as pd

df = pd.read_csv("data/processed/nfl_weekly.csv", low_memory=False)
print(df.columns[:12])
print(df.groupby('season')['week'].nunique())
print(df.query("season <= 2020")['week'].unique())  # expect [0]
print(df.query("season >= 2021")['week'].unique()[:5])  # 1..17
```

## How weeks are inferred
- For weekly files (2021–2024), week is inferred from the folder path (e.g., `NFL-data-Players/2024/1/...` → week 1).
- The cleaner also detects `week`/`wk` columns or tokens like `week01`/`wk_2` when present.

## Sharing with teammates
Point teammates to this README. They can:
- Clone the repo
- Create a virtualenv and `pip install -r requirements.txt`
- Clone `hvpkod/NFL-Data` locally
- Run the cleaner with the command above

## Regenerating outputs later
Re-run the same command after pulling updates from `NFL-Data`.

## Notes
- Parquet output requires `pyarrow`. If you don’t want Parquet, add `--no-parquet` to the run command.
- Week 18 is intentionally excluded from the weekly grid.
