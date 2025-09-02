#!/usr/bin/env python3
"""
Builds/updates data/crosswalk.db from crosswalk.csv

Usage:
  python make_crosswalk_db.py --csv crosswalk.csv --db data/crosswalk.db --rebuild
  python make_crosswalk_db.py --csv crosswalk.csv --db data/crosswalk.db
"""

import argparse
import csv
import os
import sqlite3
from pathlib import Path

import pandas as pd


# --------- helpers ---------
def detect_delimiter(csv_path: Path) -> str:
    with open(csv_path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        sample = f.read(1024 * 128)
    try:
        snif = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        return snif.delimiter
    except Exception:
        # Fall back to ';' which you used before
        return ";"


def open_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lower + strip + replace spaces/dots/“weird” chars
    def norm(c: str) -> str:
        c = c.strip().lower()
        for ch in [" ", ".", "-", "/"]:
            c = c.replace(ch, "_")
        return c

    df.columns = [norm(c) for c in df.columns]

    # Map common variants to canonical names
    mapping = {
        "tow": "tow_code",
        "tow_code": "tow_code",
        "towcode": "tow_code",
        "supplier_id": "supplier_id",
        "supplier": "supplier_id",
        "supplier_code": "supplier_id",
        "supplierid": "supplier_id",
        "vendor_id": "vendor_id",
        "vendor": "vendor_id",
        "vendorid": "vendor_id",
        "vendor_code": "vendor_id",
    }
    df = df.rename(columns={c: mapping[c] for c in df.columns if c in mapping})

    # Keep only the 3 we care about
    keep = ["tow_code", "supplier_id", "vendor_id"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected columns {keep}, but missing {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[keep]

    # Convert to string (preserve long numeric IDs)
    for c in keep:
        df[c] = df[c].astype(str).str.strip()

    # Drop blank rows
    df = df.replace({"": None})
    df = df.dropna(subset=["supplier_id", "vendor_id", "tow_code"])

    # Deduplicate on (vendor_id, supplier_id). keep last seen wins.
    df = df.drop_duplicates(subset=["vendor_id", "supplier_id"], keep="last").reset_index(drop=True)
    return df


def ensure_schema(con: sqlite3.Connection, rebuild: bool):
    cur = con.cursor()
    if rebuild:
        cur.execute("DROP TABLE IF EXISTS crosswalk;")
        cur.execute("DROP INDEX IF EXISTS ux_crosswalk_vs;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code    TEXT NOT NULL,
            supplier_id TEXT NOT NULL,
            vendor_id   TEXT NOT NULL
        );
        """
    )
    # Unique composite key for ON CONFLICT upserts
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
          ON crosswalk(vendor_id, supplier_id);
        """
    )
    con.commit()


def upsert_chunk(con: sqlite3.Connection, chunk: pd.DataFrame):
    # Perform upsert row-by-row (safe & clear; fast enough for chunked loads)
    sql = """
    INSERT INTO crosswalk(tow_code, supplier_id, vendor_id)
    VALUES (?, ?, ?)
    ON CONFLICT(vendor_id, supplier_id) DO UPDATE SET
        tow_code = excluded.tow_code
    ;
    """
    con.executemany(sql, chunk[["tow_code", "supplier_id", "vendor_id"]].itertuples(index=False, name=None))


# --------- main ---------
def main():
    ap = argparse.ArgumentParser(description="Build/Update crosswalk.db from crosswalk.csv")
    ap.add_argument("--csv", required=True, help="Path to crosswalk.csv")
    ap.add_argument("--db", default="data/crosswalk.db", help="Path to SQLite DB")
    ap.add_argument("--chunksize", type=int, default=100_000, help="Chunk size for reading CSV")
    ap.add_argument("--rebuild", action="store_true", help="Drop & recreate target table/index")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    db_path = Path(args.db)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    delimiter = detect_delimiter(csv_path)
    print(f"→ Detected delimiter: '{delimiter}'")

    con = open_connection(db_path)
    ensure_schema(con, rebuild=args.rebuild)

    total = 0
    # Try UTF-8 first, fallback to cp1250 if needed
    encodings = ["utf-8-sig", "utf-8", "cp1250"]
    read_ok = False
    for enc in encodings:
        try:
            for chunk in pd.read_csv(
                csv_path,
                dtype=str,
                chunksize=args.chunksize,
                encoding=enc,
                sep=delimiter,
                engine="python",
            ):
                chunk = normalize_columns(chunk)
                upsert_chunk(con, chunk)
                total += len(chunk)
                print(f" • chunk upserted {len(chunk):>6} rows (total {total:>7})")
            read_ok = True
            break
        except UnicodeDecodeError:
            continue

    if not read_ok:
        raise RuntimeError("Failed to read CSV in utf-8/cp1250. Please check file encoding.")

    con.commit()
    cnt = con.execute("SELECT COUNT(*) FROM crosswalk").fetchone()[0]
    con.close()
    print(f"✓ Done. Total rows in DB: {cnt} (upserted {total})")


if __name__ == "__main__":
    main()
