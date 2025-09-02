#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build (or update) SQLite crosswalk.db from a CSV.

Usage (rebuild):
  py make_crosswalk_db.py --csv crosswalk.csv --db data\crosswalk.db --rebuild

Usage (append/update-in-place):
  py make_crosswalk_db.py --csv crosswalk.csv --db data\crosswalk.db
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


# ---------- Helpers ----------

def detect_delimiter(sample_text: str, candidates: Tuple[str, ...] = (";", ",", "\t", "|")) -> str:
    """Very simple delimiter detector: pick the candidate with the most splits on the first line."""
    lines = [ln for ln in sample_text.splitlines() if ln.strip()]
    if not lines:
        return ","
    first = lines[0]
    best = ","
    best_count = -1
    for d in candidates:
        c = first.count(d)
        if c > best_count:
            best_count = c
            best = d
    return best


def ensure_schema(con: sqlite3.Connection) -> None:
    """Create table and unique index if not present."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code    TEXT,
            supplier_id TEXT,
            vendor_id   TEXT
        )
    """)
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
        ON crosswalk(vendor_id, supplier_id)
    """)
    con.commit()


# Column name normalization: accept a few common variants
CANDIDATES = {
    "tow_code": (
        "tow", "tow_code", "tow code", "product code", "towcode"
    ),
    "supplier_id": (
        "supplier_id", "supplier code", "supplier", "suppliercode", "supplier_ic", "supplier_id "
    ),
    "vendor_id": (
        "vendor_id", "vendor id", "vendor", "vendor code", "vendorid"
    ),
}


def pick_column(df: pd.DataFrame, target: str) -> str:
    """Return the actual column name in df that matches our target logical name."""
    target = target.lower()
    for col in df.columns:
        if col.strip().lower() == target:
            return col
    for alias in CANDIDATES.get(target, ()):
        for col in df.columns:
            if col.strip().lower() == alias.strip().lower():
                return col
    raise KeyError(f"Could not find column for '{target}'. Available: {list(df.columns)}")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace, coerce to string, fill NaNs."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"None": "", "nan": "", "NaN": ""})
    return df


def upsert_chunk(con: sqlite3.Connection, df: pd.DataFrame) -> None:
    """UPSERT rows using ON CONFLICT(vendor_id, supplier_id)."""
    if df.empty:
        return

    df = clean_df(df)

    col_tow = pick_column(df, "tow_code")
    col_sup = pick_column(df, "supplier_id")
    col_ven = pick_column(df, "vendor_id")

    rows: List[Tuple[str, str, str]] = list(
        zip(df[col_tow].tolist(), df[col_sup].tolist(), df[col_ven].tolist())
    )

    con.executemany(
        """
        INSERT INTO crosswalk (tow_code, supplier_id, vendor_id)
        VALUES (?, ?, ?)
        ON CONFLICT(vendor_id, supplier_id)
        DO UPDATE SET tow_code = excluded.tow_code
        """,
        rows,
    )
    # Do NOT commit here; caller decides commit cadence.


def read_csv_in_chunks(csv_path: Path, chunksize: int, sep: Optional[str]) -> Tuple[str, List[pd.DataFrame]]:
    """Yield chunks and return the chosen delimiter (for logging)."""
    if sep is None:
        # Detect delimiter on a small sample
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(8192)
        sep = detect_delimiter(sample)

    # Use engine='python' for flexible separators
    chunks = pd.read_csv(
        csv_path,
        dtype=str,
        chunksize=chunksize,
        sep=sep,
        engine="python",
    )
    return sep, chunks


# ---------- Main pipeline ----------

def build_from_csv(csv_path: Path, db_path: Path, chunksize: int, rebuild: bool) -> int:
    """Build / update the DB from CSV, return total upserted rows."""
    if rebuild and db_path.exists():
        try:
            db_path.unlink()
        except Exception as e:
            print(f"Could not remove {db_path}: {e}", file=sys.stderr)
            raise

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        ensure_schema(con)

        # Determine / detect delimiter, then stream chunks
        sep, chunk_iter = read_csv_in_chunks(csv_path, chunksize, sep=None)
        print(f"→ Detected delimiter: '{sep}'")

        total = 0
        for i, chunk in enumerate(chunk_iter, start=1):
            upsert_chunk(con, chunk)
            con.commit()
            total += len(chunk)
            print(f" • chunk {i}: upserted {len(chunk):7d} rows (total {total})")
        print(f"√ Done. Total upserted rows: {total} into {db_path}")
        return total

    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(description="Build (or update) crosswalk SQLite DB from CSV")
    ap.add_argument("--csv", required=True, help="Path to crosswalk CSV")
    ap.add_argument("--db", required=True, help="Path to output SQLite DB (e.g., data/crosswalk.db)")
    ap.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk (default: 100000)")
    ap.add_argument("--rebuild", action="store_true", help="Recreate DB from scratch")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    db_path = Path(args.db)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    total = build_from_csv(csv_path, db_path, args.chunksize, args.rebuild)

    # Simple sanity-show:
    con = sqlite3.connect(str(db_path))
    try:
        cnt = con.execute("SELECT COUNT(*) FROM crosswalk").fetchone()[0]
        print(f"DB row count now: {cnt:,}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
