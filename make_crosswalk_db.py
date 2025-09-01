#!/usr/bin/env python3
"""
Builds/updates a SQLite crosswalk DB from a CSV.

Schema:
  crosswalk(
      tow_code    TEXT,
      supplier_id TEXT,
      vendor_id   TEXT,
      UNIQUE(vendor_id, supplier_id)
  )

Usage:
  python make_crosswalk_db.py --csv crosswalk.csv --db data/crosswalk.db --rebuild
"""

from __future__ import annotations
import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def detect_delimiter(path: Path) -> str:
    """Best-effort delimiter detection; defaults to ';' then ','."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(1024)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        return dialect.delimiter
    except Exception:
        # If we see more ';' than ',', prefer ';'
        return ";" if sample.count(";") >= sample.count(",") else ","


def normalize_col(name: str) -> str:
    """Lowercase, remove spaces/underscores to compare reliably."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


# candidate sets we’ll try to map from
TOW_CANDIDATES = {
    "tow", "towcode", "towkod", "tow_kod", "tow_code", "towkod", "towkod",
}
SUPPLIER_CANDIDATES = {
    "supplierid", "supplier_id", "suppliercode", "supplier_code",
    "supplier", "sifra", "sifraean", "ean", "šifra", "šifrea n", "sifraean",
}
VENDOR_CANDIDATES = {
    "vendorid", "vendor_id", "vendor", "vendornavi", "vendornav", "vendorcode",
    "dob", "dobvendor",  # allow your DOB000025 style labels
}


def pick_column(cols: Iterable[str], candidates: set[str]) -> Optional[str]:
    norm = {normalize_col(c): c for c in cols}
    for key in candidates:
        if key in norm:
            return norm[key]
    return None


def ensure_schema(con: sqlite3.Connection, rebuild: bool = False) -> None:
    cur = con.cursor()
    if rebuild:
        cur.execute("DROP TABLE IF EXISTS crosswalk")

    # Create table (no NOT NULL so vendor_id can be blank if needed)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code    TEXT,
            supplier_id TEXT,
            vendor_id   TEXT
        )
        """
    )
    # Enforce upsert target
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vendor_supplier "
        "ON crosswalk(vendor_id, supplier_id)"
    )
    con.commit()


UPSERT_SQL = """
INSERT INTO crosswalk (tow_code, supplier_id, vendor_id)
VALUES (?, ?, ?)
ON CONFLICT(vendor_id, supplier_id) DO UPDATE
SET tow_code = excluded.tow_code
"""


def clean_cell(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def build_sqlite(csv_path: Path, db_path: Path, chunksize: int) -> None:
    delimiter = detect_delimiter(csv_path)
    print(f"→ Detected delimiter: {repr(delimiter)}")

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    # Stream the CSV to handle very large files
    chunk_iter = pd.read_csv(
        csv_path,
        engine="python",
        sep=delimiter,
        chunksize=chunksize,
        dtype=str,
    )

    total = 0
    for ix, chunk in enumerate(chunk_iter, start=1):
        # map columns
        cols = list(chunk.columns)
        tow_col = pick_column(cols, TOW_CANDIDATES)
        sup_col = pick_column(cols, SUPPLIER_CANDIDATES)
        ven_col = pick_column(cols, VENDOR_CANDIDATES)

        if not tow_col or not sup_col:
            raise KeyError(
                f"CSV missing required columns. "
                f"Need tow & supplier_id. "
                f"Found columns: {cols}"
            )
        # vendor_id can be optional; will be ""
        if ven_col is None:
            ven_col = None

        # Clean and upsert row-by-row
        rows = []
        for _, r in chunk.iterrows():
            tow = clean_cell(r[tow_col])
            sup = clean_cell(r[sup_col])
            ven = clean_cell(r[ven_col]) if ven_col else ""

            # keep only rows that have required fields
            if not sup or not tow:
                continue

            rows.append((tow, sup, ven))

        if rows:
            cur.executemany(UPSERT_SQL, rows)
            con.commit()
            total += len(rows)
            print(f"  · chunk {ix}: upserted {len(rows)} rows (total {total})")

    # Final vacuum to compact DB
    cur.execute("VACUUM")
    con.commit()
    con.close()
    print(f"✓ Done. Total upserted rows: {total} into {db_path}")


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV")
    ap.add_argument("--db", default="data/crosswalk.db", help="SQLite path (default: data/crosswalk.db)")
    ap.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk (default: 100k)")
    ap.add_argument("--rebuild", action="store_true", help="Drop & recreate schema before loading")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    db_path = Path(args.db)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(db_path))
    ensure_schema(con, rebuild=args.rebuild)
    con.close()

    build_sqlite(csv_path, db_path, args.chunksize)


if __name__ == "__main__":
    main()
