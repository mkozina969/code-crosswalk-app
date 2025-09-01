# make_crosswalk_db.py
# -----------------------------------------------------------------------------
# CLI to build or rebuild data/crosswalk.db from crosswalk.csv
# Guarantees the UNIQUE index (vendor_id, supplier_id) so the app's ON CONFLICT
# upsert works reliably.
#
# Usage examples:
#   py make_crosswalk_db.py --csv crosswalk.csv --db data/crosswalk.db --rebuild
#   py make_crosswalk_db.py --csv crosswalk.csv --db data/crosswalk.db
# -----------------------------------------------------------------------------

from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path
import pandas as pd


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # match app normalization
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    col_tow = pick(["tow", "tow_code", "towkod", "tow_kod", "towcode"]) or ("tow_code" if "tow_code" in df.columns else None)
    col_sup = pick(["supplier_id", "supplier", "vendor code", "vendor_code", "vendor code "]) or ("supplier_id" if "supplier_id" in df.columns else None)
    col_ven = pick(["vendor_id", "vendor navi", "vendor_navi", "venodr_id", "vendorid"]) or ("vendor_id" if "vendor_id" in df.columns else None)

    rename = {}
    if col_tow and col_tow != "tow_code": rename[col_tow] = "tow_code"
    if col_sup and col_sup != "supplier_id": rename[col_sup] = "supplier_id"
    if col_ven and col_ven != "vendor_id": rename[col_ven] = "vendor_id"
    if rename:
        df = df.rename(columns=rename)

    for c in ["tow_code", "supplier_id", "vendor_id"]:
        if c not in df.columns:
            df[c] = None
        df[c] = df[c].astype(str).str.strip()

    return df[["tow_code", "supplier_id", "vendor_id"]]


def build_sqlite(csv_path: Path, db_path: Path, chunksize: int = 100_000) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)

    first = True
    for i, chunk in enumerate(pd.read_csv(csv_path, dtype=str, chunksize=chunksize), start=1):
        nf = normalize(chunk)
        nf.to_sql("crosswalk", con, if_exists="replace" if first else "append", index=False)
        first = False
        print(f" • chunk {i}: upserted {len(nf):,} rows")

    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
        ON crosswalk(vendor_id, supplier_id)
    """)
    con.commit()
    con.close()
    print("√ Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to crosswalk.csv")
    ap.add_argument("--db", required=True, help="Path to data/crosswalk.db")
    ap.add_argument("--rebuild", action="store_true", help="Remove DB before build")
    ap.add_argument("--chunksize", type=int, default=100_000)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    db_path = Path(args.db)

    if args.rebuild and db_path.exists():
        db_path.unlink()

    print("→ Detected delimiter: ';' " if ';' in open(csv_path, 'r', encoding='utf-8', errors='ignore').read(1024)
          else "→ Detected delimiter: ',' (or other)")

    build_sqlite(csv_path, db_path, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
