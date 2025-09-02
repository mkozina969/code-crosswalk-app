#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build (or update) data/crosswalk.db from a CSV.

Examples
--------
# Rebuild from scratch (drops the file if present)
py make_crosswalk_db.py --csv crosswalk.csv --db data/crosswalk.db --rebuild

# Update/append (upsert) without dropping
py make_crosswalk_db.py --csv crosswalk.csv --db data/crosswalk.db
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Iterable

import pandas as pd


# ----------------------------- CLI --------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build/append SQLite crosswalk DB from CSV")
    ap.add_argument("--csv", required=True, help="Path to source CSV")
    ap.add_argument("--db", required=True, help="Path to output SQLite DB (e.g. data/crosswalk.db)")
    ap.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk (default 100k)")
    ap.add_argument("--rebuild", action="store_true", help="Drop DB file and rebuild from scratch")
    return ap.parse_args()


# -------------------------- Delimiter / Encoding ------------------------------
def detect_delimiter(sample_text: str, candidates: Tuple[str, ...] = (";", ",", "\t", "|")) -> str:
    """Pick the delimiter that appears most on the first non-empty line."""
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


def open_chunk_reader(csv_path: Path, chunksize: int, sep_hint: Optional[str] = None) -> Tuple[str, Iterable[pd.DataFrame], str]:
    """Return (delimiter, chunk_iter, encoding_used). Try utf-8, then cp1250."""
    encodings = ("utf-8-sig", "utf-8", "cp1250")
    for enc in encodings:
        with open(csv_path, "r", encoding=enc, errors="ignore") as f:
            sample = f.read(8192)
        sep = sep_hint or detect_delimiter(sample)

        try:
            itr = pd.read_csv(
                csv_path,
                dtype=str,
                chunksize=chunksize,
                sep=sep,
                engine="python",
                on_bad_lines="skip",
                encoding=enc,
            )
            # test one chunk to validate parameters; then rebuild iterator
            first = next(iter(itr))
            def new_iter():
                return pd.read_csv(
                    csv_path,
                    dtype=str,
                    chunksize=chunksize,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                    encoding=enc,
                )
            return sep, new_iter(), enc
        except StopIteration:
            return sep, iter(()), enc
        except Exception:
            continue
    raise RuntimeError("Failed to open CSV with utf-8/cp1250. Please check the file.")


# ----------------------------- Schema helpers --------------------------------
def ensure_schema(con: sqlite3.Connection) -> Tuple[bool, str]:
    """
    Ensure the crosswalk table exists.
    Returns (has_vendor, tow_col_name_to_use)
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code    TEXT,
            supplier_id TEXT NOT NULL,
            vendor_id   TEXT
        )
    """)
    # Detect actual columns (support legacy 'tow' instead of 'tow_code')
    pragma = con.execute("PRAGMA table_info(crosswalk)").fetchall()
    cols = {c[1].lower(): c[1] for c in pragma}
    tow_col = cols.get("tow_code") or cols.get("tow") or "tow_code"
    has_vendor = "vendor_id" in cols
    # Create the correct unique index
    if has_vendor:
        con.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
            ON crosswalk(vendor_id, supplier_id)
        """)
    else:
        con.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_s
            ON crosswalk(supplier_id)
        """)
    con.commit()
    return has_vendor, tow_col


def pick_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """Map CSV columns to (tow_col, supplier_col, vendor_col or None)."""
    # Normalize incoming headers
    norm = {c: c.strip().lower() for c in df.columns}
    rev = {v: k for k, v in norm.items()}

    def pick(cands: Tuple[str, ...]) -> Optional[str]:
        for c in cands:
            if c in rev:
                return rev[c]
        return None

    tow = pick(("tow_code", "tow"))
    sup = pick(("supplier_id", "supplier_code", "supplier"))
    ven = pick(("vendor_id", "vendor", "vendor_code"))

    if not tow or not sup:
        raise KeyError(
            f"Missing required columns. Need tow/tow_code AND supplier_id/supplier_code. "
            f"Found: {list(df.columns)}"
        )
    return tow, sup, ven


def normalize_chunk(df: pd.DataFrame, tow_src: str, sup_src: str, ven_src: Optional[str]) -> pd.DataFrame:
    out = pd.DataFrame({
        "tow": df[tow_src].astype(str).str.strip(),
        "supplier_id": df[sup_src].astype(str).str.strip().str.upper(),
    })
    if ven_src:
        out["vendor_id"] = df[ven_src].astype(str).str.strip().str.upper()
    else:
        out["vendor_id"] = None
    # drop rows with empty supplier_id
    out = out[out["supplier_id"] != ""]
    # deduplicate by (vendor_id, supplier_id) — keep last occurrence
    out = out.drop_duplicates(subset=["vendor_id", "supplier_id"], keep="last")
    return out


def upsert_rows(con: sqlite3.Connection, rows: pd.DataFrame, has_vendor: bool, tow_col: str) -> int:
    """UPSERT rows chunk into crosswalk."""
    if rows.empty:
        return 0
    cur = con.cursor()
    if has_vendor:
        sql = f"""
            INSERT INTO crosswalk({tow_col}, supplier_id, vendor_id)
            VALUES (?, ?, ?)
            ON CONFLICT(vendor_id, supplier_id)
            DO UPDATE SET {tow_col} = excluded.{tow_col}
        """
        data = list(rows[["tow", "supplier_id", "vendor_id"]].itertuples(index=False, name=None))
    else:
        sql = f"""
            INSERT INTO crosswalk({tow_col}, supplier_id)
            VALUES (?, ?)
            ON CONFLICT(supplier_id)
            DO UPDATE SET {tow_col} = excluded.{tow_col}
        """
        data = list(rows[["tow", "supplier_id"]].itertuples(index=False, name=None))

    cur.executemany(sql, data)
    return len(data)


# ------------------------------- Pipeline -------------------------------------
def build_from_csv(csv_path: Path, db_path: Path, chunksize: int, rebuild: bool) -> int:
    if rebuild and db_path.exists():
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")

        has_vendor, tow_col = ensure_schema(con)

        sep, chunk_iter, enc = open_chunk_reader(csv_path, chunksize)
        print(f"→ Detected delimiter: '{sep}'  | encoding: {enc}")

        total_in = 0
        total_upserted = 0

        first_chunk = True
        for i, chunk in enumerate(chunk_iter, start=1):
            if first_chunk:
                # remap columns based on the first chunk
                tow_src, sup_src, ven_src = pick_columns(chunk)
                first_chunk = False

            chunk = chunk.fillna("")
            norm = normalize_chunk(chunk, tow_src, sup_src, ven_src)
            # If file has vendor column, we keep it; else vendor_id stays None
            up = upsert_rows(con, norm, has_vendor=has_vendor, tow_col=tow_col)
            con.commit()

            total_in += len(chunk)
            total_upserted += up
            print(f" • chunk {i:>3}: read {len(chunk):>7}  | upserted {up:>7}  | total upserted {total_upserted:>9}")

        # final stats
        cnt = con.execute("SELECT COUNT(*) FROM crosswalk").fetchone()[0]
        print(f"✓ Done. CSV rows read: {total_in:,}  | upserted: {total_upserted:,}")
        print(f"  Current rows in DB: {cnt:,}")
        return int(cnt)
    finally:
        con.close()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    db_path = Path(args.db)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    build_from_csv(csv_path, db_path, args.chunksize, args.rebuild)


if __name__ == "__main__":
    main()
