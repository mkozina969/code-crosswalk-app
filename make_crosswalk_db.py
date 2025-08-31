#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a normalized SQLite crosswalk DB from a CSV exported out of SQL Server.

Resulting table schema (always):
  - tow_code     TEXT   (required)
  - supplier_id  TEXT   (required; normalized from many possible header names)
  - vendor_id    TEXT   (optional; if not present it will be NULL)

Usage:
  python make_crosswalk_db.py --csv crosswalk.csv --db crosswalk.db
  # Optional chunk size (default 200_000)
  python make_crosswalk_db.py --csv crosswalk.csv --db crosswalk.db --chunksize 300000
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Iterable, Optional

import pandas as pd


# -------------------------- helpers -------------------------- #

def detect_delimiter(csv_path: Path) -> str:
    """Detect CSV delimiter using csv.Sniffer (fallback to ';' then ',')."""
    with csv_path.open("rb") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore"))
        if dialect.delimiter in (",", ";", "\t", "|"):
            print(f"→ Detected delimiter via Sniffer: {repr(dialect.delimiter)}")
            return dialect.delimiter
    except Exception:
        pass

    # quick heuristic: pick the one that appears more often in the header line
    header = sample.splitlines()[0].decode("utf-8", errors="ignore")
    semi = header.count(";")
    comma = header.count(",")
    if semi >= comma:
        print("→ Fallback delimiter chosen: ';'")
        return ";"
    print("→ Fallback delimiter chosen: ','")
    return ","


def norm_header(s: str) -> str:
    """Normalize a column header for matching."""
    return (
        s.strip()
         .lower()
         .replace("š", "s").replace("č", "c").replace("ć", "c")
         .replace("đ", "d").replace("ž", "z")
         .replace(" ", "_")
         .replace("-", "_")
         .replace(".", "_")
         .replace("__", "_")
    )


# Accept many possible header variants the user might export from SQL/Excel
TOW_ALIASES: tuple[str, ...] = (
    "tow", "tow_code", "towkod", "tow_kod", "tow_kod_as_tow", "tow_kod_as_towkod"
)

SUPPLIER_ALIASES: tuple[str, ...] = (
    "supplier_id", "supplier", "supplier_code", "suppliercode", "sup_code",
    "suppler_code", "supplerid",
    "vendor_code", "vendor code", "vendor",  # some users export vendor code as supplier code
    "sifra", "sifra_proizvoda", "sifra_ean", "sifra_ean13"  # if you ever remap from these
)

VENDOR_ALIASES: tuple[str, ...] = (
    "vendor_id", "vendor", "vendor_navi", "vendor_navi_id", "vendornavi", "vendorid",
    "vendor__id", "vendor_id_", "ven_id"
)


def pick_first(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Return the first column from cols that matches any candidate."""
    cols_norm = {norm_header(c): c for c in cols}
    for cand in candidates:
        key = norm_header(cand)
        if key in cols_norm:
            return cols_norm[key]
        # also allow exact candidate present already normalized
        if cand in cols_norm:
            return cols_norm[cand]
    return None


def clean_string_series(s: pd.Series) -> pd.Series:
    """Return string series upper-stripped w/o surrounding quotes."""
    s = s.astype("string", errors="ignore")
    s = s.fillna("").astype(str).str.strip().str.strip('"').str.strip("'")
    # don't force upper for numeric-like strings; we keep original but remove spaces
    return s


# -------------------------- core build -------------------------- #

def build_sqlite(csv_path: Path, db_path: Path, chunksize: int = 200_000) -> None:
    """(Re)build the crosswalk SQLite DB from a CSV."""
    if db_path.exists():
        print(f"🧹 Removing existing DB: {db_path}")
        db_path.unlink()

    sep = detect_delimiter(csv_path)
    print(f"📥 Reading CSV in chunks (sep={repr(sep)}, chunksize={chunksize:,}) …")

    # Open DB and create table
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS crosswalk")
    cur.execute(
        """
        CREATE TABLE crosswalk (
            tow_code     TEXT,
            supplier_id  TEXT,
            vendor_id    TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS ix_crosswalk_tow ON crosswalk(tow_code)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_crosswalk_supplier ON crosswalk(supplier_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_crosswalk_vendor ON crosswalk(vendor_id)")
    con.commit()

    # Chunked load
    total_rows = 0
    chunk_iter = pd.read_csv(
        csv_path,
        sep=sep,
        dtype=str,
        header=0,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunksize,
        on_bad_lines="skip",
        encoding="utf-8",
        engine="python",
    )

    for i, df in enumerate(chunk_iter, start=1):
        # Normalize headers
        cols_map = {c: norm_header(c) for c in df.columns}
        df.rename(columns=cols_map, inplace=True)

        tow_col = pick_first(df.columns, TOW_ALIASES)
        sup_col = pick_first(df.columns, SUPPLIER_ALIASES)
        ven_col = pick_first(df.columns, VENDOR_ALIASES)

        missing = []
        if not tow_col:
            missing.append("TOW (any of: " + ", ".join(TOW_ALIASES) + ")")
        if not sup_col:
            missing.append("SUPPLIER_ID (any of: " + ", ".join(SUPPLIER_ALIASES) + ")")

        if missing:
            con.close()
            raise ValueError(
                "Required columns not found in CSV chunk.\nMissing: "
                + "; ".join(missing)
                + f"\nColumns in file: {list(df.columns)}"
            )

        # Create normalized columns
        out = pd.DataFrame({
            "tow_code":     clean_string_series(df[tow_col]),
            "supplier_id":  clean_string_series(df[sup_col]),
            "vendor_id":    clean_string_series(df[ven_col]) if ven_col else "",
        })

        # Drop completely empty rows (just in case)
        out = out[
            out["tow_code"].astype(str).str.len().gt(0)
            | out["supplier_id"].astype(str).str.len().gt(0)
        ]

        # Append to SQLite
        out.to_sql("crosswalk", con, if_exists="append", index=False)
        total_rows += len(out)
        print(f"  • chunk {i:>4} → inserted {len(out):>8,} rows (cumulative {total_rows:>10,})")

    con.commit()
    con.close()
    print(f"✅ Finished building DB: {db_path}  (rows written: {total_rows:,})")

    # Copy to ./data/crosswalk.db for Streamlit to auto-load
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    target = data_dir / "crosswalk.db"
    shutil.copy2(db_path, target)
    print(f"📦 Copied DB to {target.resolve()}")


# -------------------------- CLI -------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Build normalized crosswalk SQLite DB")
    ap.add_argument("--csv", required=True, help="Input CSV file")
    ap.add_argument("--db", default="crosswalk.db", help="SQLite DB output path")
    ap.add_argument(
        "--chunksize", type=int, default=200_000,
        help="Rows per chunk while reading CSV (default: 200k)"
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    db_path = Path(args.db)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"🏗  Building SQLite from: {csv_path.resolve()}")
    build_sqlite(csv_path, db_path, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
