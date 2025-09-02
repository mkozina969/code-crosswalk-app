#!/usr/bin/env python3
"""
Build (or rebuild) the crosswalk SQLite DB from a CSV.

Expected logical columns (case/spacing doesn't matter; common aliases supported):
- tow_code       -> "tow", "tow_code", "towkod", "tow_kod"
- supplier_id    -> "supplier", "supplier_id", "sup_code", "sup_id"
- vendor_id      -> "vendor", "vendor_id", "vend", "vend_id", "partner", "dob"

Creates:
  data/crosswalk.db  (or a custom --db path)

Schema:
  CREATE TABLE IF NOT EXISTS crosswalk (
      tow_code    TEXT NOT NULL,
      supplier_id TEXT NOT NULL,
      vendor_id   TEXT NOT NULL
  );

  CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
    ON crosswalk(vendor_id, supplier_id);

Usage:
  py make_crosswalk_db.py --csv crosswalk.csv --db data\crosswalk.db --rebuild
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build crosswalk SQLite DB from CSV")
    p.add_argument("--csv", required=True, help="Path to crosswalk CSV")
    p.add_argument("--db", default="data/crosswalk.db", help="Path to SQLite DB to write")
    p.add_argument("--chunksize", type=int, default=100_000, help="Pandas chunk size")
    p.add_argument("--rebuild", action="store_true", help="Delete DB first and rebuild")
    return p.parse_args()


# -----------------------------
# Utilities
# -----------------------------
CANDIDATE_TOW = ("tow", "tow_code", "towkod", "tow_kod", "tow code", "tow-code")
CANDIDATE_SUP = ("supplier_id", "supplier", "sup_code", "sup_id", "supplier id", "supplier-id")
CANDIDATE_VEN = ("vendor_id", "vendor", "vend", "vend_id", "partner", "dob", "dob#")

def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
        delim = dialect.delimiter
    except Exception:
        # Fallback: guess ; then ,
        delim = ";" if sample.count(";") >= sample.count(",") else ","
    print(f"→ Detected delimiter: {repr(delim)}")
    return delim


def pick_col(cols: Iterable[str], candidates: Tuple[str, ...]) -> Optional[str]:
    norm = {c.strip().lower(): c for c in cols}
    for cand in candidates:
        key = cand.lower()
        if key in norm:
            return norm[key]
    return None


def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    # lower trimmed column names
    df.columns = [c.strip().lower() for c in df.columns]

    tow = pick_col(df.columns, CANDIDATE_TOW)
    sup = pick_col(df.columns, CANDIDATE_SUP)
    ven = pick_col(df.columns, CANDIDATE_VEN)

    missing = []
    if tow is None:
        missing.append("tow_code")
    if sup is None:
        missing.append("supplier_id")
    if ven is None:
        missing.append("vendor_id")

    if missing:
        raise KeyError(f"Missing required logical columns in CSV: {', '.join(missing)}")

    out = df[[tow, sup, ven]].copy()
    out.columns = ["tow_code", "supplier_id", "vendor_id"]

    # Clean strings
    for c in ("tow_code", "supplier_id", "vendor_id"):
        out[c] = out[c].astype(str).str.strip()

    # Drop blank required fields
    out = out[(out["supplier_id"] != "") & (out["vendor_id"] != "") & (out["tow_code"] != "")]
    # Drop duplicates within the chunk to avoid unique-index collisions
    out = out.drop_duplicates(subset=["vendor_id", "supplier_id"], keep="last")

    return out


def ensure_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code    TEXT NOT NULL,
            supplier_id TEXT NOT NULL,
            vendor_id   TEXT NOT NULL
        );
        """
    )
    con.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
        ON crosswalk(vendor_id, supplier_id);
        """
    )
    con.commit()


# -----------------------------
# Main build
# -----------------------------
def build(db_path: Path, csv_path: Path, chunksize: int) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    delim = detect_delimiter(csv_path)

    # Use context manager so the connection is cleanly closed
    with sqlite3.connect(str(db_path)) as con:
        ensure_schema(con)

        total = 0
        # Engine='python' is safer with odd delimiters; on_bad_lines='skip' tolerates stray lines
        for i, chunk in enumerate(
            pd.read_csv(
                csv_path,
                sep=delim,
                dtype=str,
                chunksize=chunksize,
                engine="python",
                keep_default_na=False,
                na_filter=False,
                on_bad_lines="skip",
            ),
            start=1,
        ):
            normalized = normalize_chunk(chunk)

            # We *append* rows; unique index prevents duplicates across the whole table.
            # To avoid IntegrityError across-chunk, de-dup vs existing by left join in SQL is costly.
            # Instead, we try insert and if it fails, we drop duplicates by replacing chunk rows
            # colliding with existing. The simplest/fast approach is to try 'append' and
            # ignore duplicates per chunk via drop_duplicates (already done).
            try:
                normalized.to_sql("crosswalk", con, if_exists="append", index=False)
            except sqlite3.IntegrityError:
                # As a fallback, upsert row-by-row for only conflicting subset
                # (rare path if CSV had duplicates spanning different chunks).
                # Use executemany with ON CONFLICT DO UPDATE:
                con.executemany(
                    """
                    INSERT INTO crosswalk(tow_code, supplier_id, vendor_id)
                    VALUES(?, ?, ?)
                    ON CONFLICT(vendor_id, supplier_id) DO UPDATE
                    SET tow_code=excluded.tow_code
                    """,
                    list(map(tuple, normalized[["tow_code", "supplier_id", "vendor_id"]].values)),
                )

            total += len(normalized)
            print(f"  • chunk {i}: upserted {len(normalized):>6} rows (total {total:>7})")

        con.commit()

    print(f"✓ Done. Total upserted rows: {total} into {db_path}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    db_path = Path(args.db)

    if args.rebuild and db_path.exists():
        try:
            db_path.unlink()
            print(f"⚡ Rebuild requested: deleted existing DB {db_path}")
        except Exception as e:
            print(f"Could not delete {db_path}: {e}")

    build(db_path, csv_path, args.chunksize)


if __name__ == "__main__":
    main()
