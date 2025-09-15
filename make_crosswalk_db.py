#!/usr/bin/env python3
"""
Build/refresh the SQLite crosswalk DB from a CSV.

- Accepts flexible headers:
    * tow code:  "TOW" / "tow" / "tow_code" / "towcode" / etc.
    * supplier:  "supplier_id" / "supplier_code" / "supplier" / "supplier_ic" / "supplieric"
    * vendor:    "vendor_id" / "vendor" / "vendor_code"  (optional)

- Always stores in DB columns: (tow_code, supplier_id, vendor_id)
- Creates UNIQUE index on (vendor_id, supplier_id).
- Upserts via ON CONFLICT to keep the latest tow_code for a pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import sqlite3


# ----------------------------
# CSV helpers
# ----------------------------
def detect_sep_and_skip(path: Path, enc: str = "utf-8") -> Tuple[str, int]:
    """Detect separator (comma/semicolon) and whether first line is 'sep=;'."""
    head = path.read_bytes()[:4096].decode(enc, errors="ignore")
    first = (head.splitlines() or [""])[0].strip().lower()
    sep = ";" if first.count(";") > first.count(", ") else ", "
    skip = 1 if first.startswith("sep=") else 0
    return sep, skip


def read_csv_iter(
    csv_path: Path,
    chunksize: int,
    preferred_encoding: str = "utf-8",
):
    """
    Robust CSV reader yielding pandas chunks with dtype=str (to avoid scientific notation).
    Tries utf-8 then latin-1; auto-detects sep; skips 'sep=;' first row if present.
    """
    for enc in (preferred_encoding, "latin-1"):
        try:
            sep, skip = detect_sep_and_skip(csv_path, enc)
            return pd.read_csv(
                csv_path,
                sep=sep,
                engine="python",
                dtype=str,
                encoding=enc,
                on_bad_lines="skip",
                chunksize=chunksize,
                skiprows=skip,
            )
        except Exception:
            continue
    # If both encodings fail, re-raise utf-8 attempt
    sep, skip = detect_sep_and_skip(csv_path, preferred_encoding)
    return pd.read_csv(
        csv_path,
        sep=sep,
        engine="python",
        dtype=str,
        encoding=preferred_encoding,
        on_bad_lines="skip",
        chunksize=chunksize,
        skiprows=skip,
    )


# ----------------------------
# Column mapping & normalization
# ----------------------------
def pick_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """
    Map CSV columns to (tow_col, supplier_col, vendor_col or None).
    Accepts uppercase TOW and common variations.
    """
    # build lowercase->original map
    low_map = {c.strip().lower(): c for c in df.columns}
    # rev = {k: v for v, k in low_map.items()}  # original->lower (unused)

    def pick(cands) -> Optional[str]:
        for c in cands:
            if c in low_map:
                return low_map[c]
        return None

    tow = pick(("tow_code", "tow", "towid", "towidcode", "towcode", "tow_"))
    if not tow:
        # explicit pass for uppercase TOW as in your screenshot
        for c in df.columns:
            if c.strip().upper() == "TOW":
                tow = c
                break

    supplier = pick(("supplier_id", "supplier_code", "supplier", "supplieric", "supplier_ic"))

    vendor = pick(("vendor_id", "vendor", "vendor_code"))

    if not tow or not supplier:
        raise KeyError(
            f"Missing required columns. Need tow/tow_code AND supplier_id/supplier_code. "
            f"Found: {list(df.columns)}"
        )
    return tow, supplier, vendor


def normalize_chunk(df: pd.DataFrame, tow_src: str, sup_src: str, ven_src: Optional[str]) -> pd.DataFrame:
    """
    Normalize a raw CSV chunk to canonical columns and clean types/whitespace.
    Avoids scientific notation by keeping everything as strings.
    """
    # supplier_id may be numeric-looking → ensure plain string (no sci notation)
    def safe_sup(x) -> str:
        s = "" if x is None else str(x)
        s = s.strip()
        # Handle common Excel-ish artifacts
        if s.endswith(".0") and s.replace(".", "", 1).isdigit():
            s = s[:-2]
        return s.upper()

    out = pd.DataFrame(
        {
            "tow_code": df[tow_src].astype(str).str.strip(),
            "supplier_id": df[sup_src].map(safe_sup),
        }
    )
    if ven_src:
        out["vendor_id"] = df[ven_src].astype(str).str.strip().str.upper()
    else:
        out["vendor_id"] = None

    # drop empty supplier_ids and keep last duplicate per (vendor_id, supplier_id)
    out = out[out["supplier_id"] != ""]
    out = out.drop_duplicates(subset=["vendor_id", "supplier_id"], keep="last")
    return out


# ----------------------------
# DB helpers
# ----------------------------
def ensure_schema(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code    TEXT NOT NULL,
            supplier_id TEXT NOT NULL,
            vendor_id   TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_crosswalk_vendor_supplier
        ON crosswalk (vendor_id, supplier_id)
        """
    )
    con.commit()


def write_chunk(con: sqlite3.Connection, df: pd.DataFrame):
    """
    Upsert a normalized chunk into DB.
    """
    if df.empty:
        return
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO crosswalk (tow_code, supplier_id, vendor_id)
        VALUES (?, ?, ?)
        ON CONFLICT(vendor_id, supplier_id)
        DO UPDATE SET tow_code = excluded.tow_code
        """,
        df[["tow_code", "supplier_id", "vendor_id"]].itertuples(index=False, name=None),
    )
    con.commit()


# ----------------------------
# Build process
# ----------------------------
def build_from_csv(csv_path: Path, db_path: Path, chunksize: int, rebuild: bool):
    total_rows = 0

    if rebuild and db_path.exists():
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        ensure_schema(con)

        # read one small chunk first to decide columns
        first_iter = read_csv_iter(csv_path, chunksize=chunksize)
        first_chunk = next(iter(first_iter))
        tow_col, sup_col, ven_col = pick_columns(first_chunk)
        norm = normalize_chunk(first_chunk, tow_col, sup_col, ven_col)
        write_chunk(con, norm)
        total_rows += len(norm)
        print(f"• chunk 1: upserted {len(norm):6d} rows (total {total_rows:7d})")

        # continue for remaining chunks with a fresh iterator
        rest_iter = read_csv_iter(csv_path, chunksize=chunksize)
        chunk_no = 1
        for chunk in rest_iter:
            chunk_no += 1
            norm = normalize_chunk(chunk, tow_col, sup_col, ven_col)
            write_chunk(con, norm)
            total_rows += len(norm)
            print(f"• chunk {chunk_no}: upserted {len(norm):6d} rows (total {total_rows:7d})")

    finally:
        con.close()

    print(f"✓ Done. Total upserted rows: {total_rows:, } into {db_path}")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Create/refresh data/crosswalk.db from a CSV.")
    ap.add_argument("--csv", required=True, type=Path, help="Path to crosswalk CSV")
    ap.add_argument("--db", default=Path("data/crosswalk.db"), type=Path, help="Output SQLite DB")
    ap.add_argument("--chunksize", default=100_000, type=int, help="Rows per chunk")
    ap.add_argument("--rebuild", action="store_true", help="Delete DB first and rebuild")
    return ap.parse_args()


def main():
    args = parse_args()
    print("→ Detected delimiter: auto ; or ,  (first line 'sep=…' supported)")
    build_from_csv(args.csv, args.db, args.chunksize, args.rebuild)


if __name__ == "__main__":
    main()
