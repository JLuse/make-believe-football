import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


SNAKE_CASE_PATTERN = re.compile(r"[^a-zA-Z0-9]+")
EXCLUDED_POSITIONS = {"DB", "DL", "LB"}
VALID_WEEK_MIN = 1
VALID_WEEK_MAX = 23  # inclusive; allows regular + some postseason weeks
WEEKLY_SEASON_MIN = 2021
WEEKLY_SEASON_MAX = 2024
TOTALS_SEASON_MIN = 2015
TOTALS_SEASON_MAX = 2020


@dataclass
class SourceFileMetadata:
    season: Optional[int]
    position: Optional[str]
    week: Optional[int]
    source_path: Path


def to_snake_case(name: str) -> str:
    lowered = name.strip().lower()
    replaced = SNAKE_CASE_PATTERN.sub("_", lowered)
    normalized = re.sub(r"_+", "_", replaced).strip("_")
    return normalized


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(str(col)) for col in df.columns]
    return df


WEEK_TOKEN_REGEX = re.compile(r"(?i)(?:^|[^a-z0-9])(?:wk|week)[ _-]?(\d{1,2})(?:[^a-z0-9]|$)")


def extract_season_and_position(path: Path) -> SourceFileMetadata:
    season: Optional[int] = None
    position: Optional[str] = None
    week: Optional[int] = None

    # Heuristic: repo layout resembles NFL-data-Players/<season>/<position>/...
    parts = [p for p in path.parts]
    # Try to find the first 4-digit year in the path
    for part in parts:
        if re.fullmatch(r"20\d{2}", part):
            try:
                season = int(part)
            except ValueError:
                pass
    # Position is likely a directory name like QB/RB/WR/TE/K/DST
    known_positions = {"qb", "rb", "wr", "te", "k", "dst", "def", "pk"}
    for part in parts:
        if part.lower() in known_positions:
            position = part.upper()
    # Try to detect week from plain numeric directories like .../<season>/<week>/
    for part in parts:
        if re.fullmatch(r"\d{1,2}", part or ""):
            try:
                w_num = int(part)
                if 1 <= w_num <= 22:
                    week = w_num
            except ValueError:
                pass
    # Try to detect week from any part or filename stem
    for token in parts + [path.stem]:
        match = WEEK_TOKEN_REGEX.search(token)
        if match:
            try:
                w = int(match.group(1))
                if 1 <= w <= 22:  # include preseason/postseason margin
                    week = w
                    break
            except ValueError:
                pass
    return SourceFileMetadata(season=season, position=position, week=week, source_path=path)


def standardize_position(raw_position: Optional[str]) -> Optional[str]:
    if raw_position is None:
        return None
    pos = raw_position.strip().upper()
    mapping = {
        "DEF": "DST",
        "D/ST": "DST",
        "PK": "K",
    }
    return mapping.get(pos, pos)


def standardize_team_abbr(team: Optional[str], season: Optional[int]) -> Optional[str]:
    if team is None:
        return None
    abbr = team.strip().upper()

    # Historical relocations/renames with season-sensitive mapping
    # STL->LAR (2016), SD->LAC (2017), OAK->LV (2020)
    if abbr == "STL" and (season is None or season >= 2016):
        return "LAR"
    if abbr == "SD" and (season is None or season >= 2017):
        return "LAC"
    if abbr == "OAK" and (season is None or season >= 2020):
        return "LV"

    # Normalize common alt spellings
    common = {
        "JAX": "JAC",
        "WSH": "WAS",
        "LA": "LAR",  # ambiguous; assume post-2016 Rams
    }
    return common.get(abbr, abbr)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    do_not_convert = {
        "player", "player_name", "name", "team", "opponent", "opp",
        "position", "pos", "status", "note", "notes"
    }
    for col in df.columns:
        if col in do_not_convert:
            continue
        # Skip columns that look categorical
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="ignore")
            df[col] = converted
    return df


def filter_excluded_positions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "position" not in df.columns:
        return df
    mask = ~df["position"].astype(str).str.upper().isin(EXCLUDED_POSITIONS)
    return df.loc[mask].reset_index(drop=True)


def filter_valid_weeks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "week" not in df.columns:
        # No explicit/inferred week → keep for now (may still be useful)
        return df
    weeks = pd.to_numeric(df["week"], errors="coerce")
    mask = (weeks >= VALID_WEEK_MIN) & (weeks <= VALID_WEEK_MAX)
    return df.loc[mask].reset_index(drop=True)


def filter_seasons_window(df: pd.DataFrame, start_season: int, end_season: int) -> pd.DataFrame:
    df = df.copy()
    if "season" not in df.columns:
        return df
    seasons = pd.to_numeric(df["season"], errors="coerce")
    return df.loc[(seasons >= start_season) & (seasons <= end_season)].reset_index(drop=True)


def apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Map variations to a canonical weekly stat schema
    aliases = {
        # identity columns
        "playername": "player",
        "player_id": "player_id",
        "playerid": "player_id",
        "pos": "position",
        # passing
        "passingyds": "pass_yds",
        "passingtd": "pass_td",
        "passingint": "pass_int",
        # rushing
        "rushingyds": "rush_yds",
        "rushingtd": "rush_td",
        # receiving
        "receivingrec": "rec",
        "receivingyds": "rec_yds",
        "receivingtd": "rec_td",
        # misc
        "rettd": "ret_td",
        "fumtd": "fum_td",
        "2pt": "two_pt",
        "fum": "fumbles",
    }
    rename_map = {c: aliases[c] for c in df.columns if c in aliases}
    if rename_map:
        df = df.rename(columns=rename_map)
    # Ensure a single canonical player column
    if "player" not in df.columns:
        for candidate in ("playername", "name"):
            if candidate in df.columns:
                df.rename(columns={candidate: "player"}, inplace=True)
                break
    return df


def drop_numeric_named_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keep_mask = ~df.columns.to_series().astype(str).str.fullmatch(r"\d+")
    return df.loc[:, list(df.columns[keep_mask])]


def parse_week_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for candidate in ("week", "wk"):
        if candidate in df.columns:
            df["week"] = pd.to_numeric(df[candidate], errors="coerce").astype("Int64")
            break
    return df


def infer_week_from_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "week" in df.columns:
        return df
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        return df
    extracted: Optional[pd.Series] = None
    for col in text_cols:
        series = df[col].astype(str)
        matches = series.str.extract(WEEK_TOKEN_REGEX, expand=False)
        if matches.notna().any():
            nums = pd.to_numeric(matches, errors="coerce")
            if nums.notna().any():
                extracted = nums.astype("Int64")
                break
    if extracted is not None and "week" not in df.columns:
        df["week"] = extracted
    return df


def attach_metadata_columns(df: pd.DataFrame, meta: SourceFileMetadata) -> pd.DataFrame:
    df = df.copy()
    if "season" not in df.columns and meta.season is not None:
        df["season"] = meta.season
    # Position from metadata has priority if present
    if meta.position is not None:
        df["position"] = standardize_position(meta.position)
    else:
        # Try to standardize if present in data
        if "position" in df.columns:
            df["position"] = df["position"].apply(lambda v: standardize_position(v))
        elif "pos" in df.columns:
            df["position"] = df["pos"].apply(lambda v: standardize_position(v))
    # Add week from metadata if present and not already set
    if "week" not in df.columns and meta.week is not None:
        df["week"] = meta.week
    # Standardize teams
    for team_col in ("team", "player_team", "club"):
        if team_col in df.columns:
            df["team"] = df[team_col].apply(lambda v: standardize_team_abbr(v, df.get("season", pd.Series([None] * len(df))).iloc[0] if len(df) else None))
            break
    if "opp" in df.columns and "opponent" not in df.columns:
        df["opponent"] = df["opp"].apply(lambda v: standardize_team_abbr(v, df.get("season", pd.Series([None] * len(df))).iloc[0] if len(df) else None))
    elif "opponent" in df.columns:
        df["opponent"] = df["opponent"].apply(lambda v: standardize_team_abbr(v, df.get("season", pd.Series([None] * len(df))).iloc[0] if len(df) else None))
    return df


def read_csv_file(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        # Fallback to python engine and permissive settings
        return pd.read_csv(path, engine="python", encoding_errors="ignore")


def read_json_file(path: Path) -> pd.DataFrame:
    # Try JSON lines first
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        # Try reading as a list
        with path.open("r", encoding="utf-8") as f:
            text = f.read().strip()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Fall back to line-by-line objects
                records = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                data = records
        return pd.DataFrame(data)


def iter_source_files(source_dir: Path) -> Iterable[Tuple[SourceFileMetadata, pd.DataFrame]]:
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".json"}:
            continue
        meta = extract_season_and_position(path)
        try:
            if path.suffix.lower() == ".csv":
                df = read_csv_file(path)
            else:
                df = read_json_file(path)
        except Exception as ex:
            print(f"Skipping unreadable file: {path} ({ex})")
            continue
        yield meta, df


def clean_one_dataframe(df: pd.DataFrame, meta: SourceFileMetadata) -> pd.DataFrame:
    df = normalize_column_names(df)
    # Ensure unique column names early to avoid concat alignment errors later
    if not pd.Index(df.columns).is_unique:
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df = attach_metadata_columns(df, meta)
    df = filter_excluded_positions(df)
    df = parse_week_column(df)
    df = infer_week_from_text_columns(df)
    df = filter_valid_weeks(df)
    if df.empty:
        return df
    df = apply_column_aliases(df)
    df = coerce_numeric_columns(df)
    df = drop_numeric_named_columns(df)

    # Standardize player name column
    if "player" not in df.columns:
        for candidate in ("player_name", "name", "playerid", "player_id"):
            if candidate in df.columns:
                df.rename(columns={candidate: "player"}, inplace=True)
                break

    # Reorder commonly used columns first when present
    preferred_order = [
        "season", "week", "player", "team", "position", "opponent"
    ]
    existing = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    df = df[existing + remaining]
    return df


def consolidate(source_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for meta, raw_df in iter_source_files(source_dir):
        if raw_df is None or raw_df.shape[0] == 0:
            continue
        cleaned = clean_one_dataframe(raw_df, meta)
        if cleaned.shape[0] == 0:
            continue
        # Double-check for any duplicate-named columns
        if not pd.Index(cleaned.columns).is_unique:
            cleaned = cleaned.loc[:, ~pd.Index(cleaned.columns).duplicated()]
        frames.append(cleaned)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Remove any existing week 18 rows pre-emptively
    if "week" in combined.columns:
        combined = combined.loc[~combined["week"].eq(18)].reset_index(drop=True)

    # Split into totals-era and weekly-era
    weekly_part = filter_seasons_window(combined, WEEKLY_SEASON_MIN, WEEKLY_SEASON_MAX)
    totals_part = filter_seasons_window(combined, TOTALS_SEASON_MIN, TOTALS_SEASON_MAX)

    # For weekly era, ensure complete weekly grid
    if not weekly_part.empty:
        weekly_part = expand_complete_weeks(weekly_part)

    # For totals era, collapse to season totals per player only
    if not totals_part.empty and {"season", "player"}.issubset(totals_part.columns):
        numeric_cols = totals_part.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["season", "week"]]
        totals = totals_part.copy()
        if "week" in totals.columns:
            totals = totals.drop(columns=["week"])  # ignore weekly dimension in totals era
        grouped = totals.groupby(["season", "player"], as_index=False)[numeric_cols].sum(min_count=1)
        # carry latest team/position in that season
        meta = (
            totals_part.sort_values(["season", "player"] + (["week"] if "week" in totals_part.columns else []), na_position="last")
                       .drop_duplicates(subset=["season", "player"], keep="last")
                       [["season", "player", "team", "position"]]
        )
        totals_result = grouped.merge(meta, on=["season", "player"], how="left")
        totals_result["week"] = 0
        # Reorder
        preferred = ["season", "week", "player", "team", "position"]
        other_cols = [c for c in grouped.columns if c not in ["season", "player"]]
        for c in other_cols:
            if c not in preferred:
                preferred.append(c)
        totals_result = totals_result.reindex(columns=preferred + [c for c in combined.columns if c not in preferred])

    else:
        totals_result = pd.DataFrame(columns=["season", "week", "player", "team", "position"])  # empty

    combined = pd.concat([weekly_part, totals_result], ignore_index=True, sort=False)

    # Keep schema consistent and ordered
    preferred_order = ["season", "week", "player", "team", "position", "opponent"]
    existing = [c for c in preferred_order if c in combined.columns]
    remaining = [c for c in combined.columns if c not in existing]
    combined = combined[existing + remaining]

    # Basic dedupe
    dedupe_keys = [k for k in ["season", "week", "player", "team", "position"] if k in combined.columns]
    if dedupe_keys:
        combined = combined.drop_duplicates(subset=dedupe_keys)

    # Append season totals per player (sum of numeric stats across weeks)
    if {"season", "week", "player"}.issubset(combined.columns):
        numeric_cols = combined.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["season", "week"]]
        if numeric_cols:
            weekly = combined.dropna(subset=["week"])  # only weekly rows
            totals = weekly.groupby(["season", "player"], as_index=False)[numeric_cols].sum(min_count=1)

            # carry latest team/position within season
            last_meta = (
                weekly.sort_values(["season", "player", "week"], na_position="last")
                      .drop_duplicates(subset=["season", "player"], keep="last")
                      [["season", "player", "team", "position"]]
            )
            totals = totals.merge(last_meta, on=["season", "player"], how="left")
            totals["week"] = 0  # denote season totals with week 0

            # align column order; add any missing columns as NA
            order = ["season", "week", "player", "team", "position"]
            for c in combined.columns:
                if c not in order:
                    order.append(c)
            totals = totals.reindex(columns=order)
            combined = pd.concat([combined, totals], ignore_index=True, sort=False)

    return combined


def save_outputs(df: pd.DataFrame, output_dir: Path, write_csv: bool = True, write_parquet: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_parquet:
        try:
            df.to_parquet(output_dir / "nfl_weekly.parquet", index=False)
        except Exception as ex:
            print(f"Failed to write parquet ({ex}); continuing with CSV if enabled.")
    if write_csv:
        df.to_csv(output_dir / "nfl_weekly.csv", index=False)


def expand_complete_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Create a complete grid of (season, week, player) for seasons >= 2020.
    Missing rows are filled with NA stats while preserving identity columns.
    """
    if df.empty:
        return df
    required_identity = ["season", "player", "team", "position"]
    for col in required_identity:
        if col not in df.columns:
            # Cannot expand without core identity
            return df

    # Restrict weeks 1..17 for regular season completeness (exclude week 18)
    all_weeks = pd.Index(range(1, 18), dtype=int)

    # Distinct players per season with latest known team/position that season
    id_cols = ["player", "team", "position"]
    sort_keys = ["season", "player"] + (["week"] if "week" in df.columns else [])
    latest = (df.sort_values(sort_keys, na_position="last")
                .drop_duplicates(subset=["season", "player"], keep="last")
                [["season"] + id_cols])

    season_week = (
        latest.assign(key=1)
              .merge(pd.DataFrame({"week": all_weeks, "key": 1}), on="key")
              .drop(columns=["key"]) 
    )

    # Left-join original stats by (season, week, player) only
    # Ensure a week column exists in df for the merge; if absent, create empty
    if "week" not in df.columns:
        df = df.assign(week=pd.Series(dtype="Int64"))

    # Coalesce multiple rows per player-week across sources: numeric = max, text = first non-null
    df_tmp = df.copy()
    for col in df_tmp.columns:
        if col not in ["season", "week", "player"] and df_tmp[col].dtype == object:
            df_tmp[col] = df_tmp[col].replace("", pd.NA)

    numeric_cols = [c for c in df_tmp.select_dtypes(include=["number"]).columns 
                    if c not in ["season", "week"]]
    other_cols = [c for c in df_tmp.columns if c not in ["season", "week", "player"] + numeric_cols]

    def first_non_null(series: pd.Series):
        non_null = series.dropna()
        return non_null.iloc[0] if len(non_null) else pd.NA

    agg_dict = {col: "max" for col in numeric_cols}
    for col in other_cols:
        agg_dict[col] = first_non_null

    df_stats = (
        df_tmp.groupby(["season", "week", "player"], as_index=False)
              .agg(agg_dict)
    )

    # Do not require team/position to match when merging stats
    stat_cols = [c for c in df_stats.columns if c not in ["season", "week", "player", "team", "position"]]
    filled = (
        season_week.merge(df_stats[["season", "week", "player"] + stat_cols],
                          on=["season", "week", "player"],
                          how="left")
    )
    # Keep column order
    preferred_order = ["season", "week", "player", "team", "position", "opponent"]
    existing = [c for c in preferred_order if c in filled.columns]
    remaining = [c for c in filled.columns if c not in existing]
    return filled[existing + remaining]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and consolidate NFL-Data repository files.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=os.environ.get("NFL_DATA_SOURCE", ""),
        help="Path to local clone of hvpkod/NFL-Data (e.g., .../NFL-Data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to write consolidated outputs",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable writing CSV output",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Disable writing Parquet output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_dir:
        raise SystemExit("--source-dir is required (or set NFL_DATA_SOURCE env var)")
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    df = consolidate(source_dir)
    if df.empty:
        print("No data found to process.")
        return
    save_outputs(df, output_dir, write_csv=not args.no_csv, write_parquet=not args.no_parquet)
    print(f"Rows consolidated: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()


