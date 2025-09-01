import argparse
import pandas as pd
import sqlite3
from pathlib import Path

def rebuild_db(csv_path: Path, db_path: Path, chunksize: int = 100_000):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con, con:
        con.execute("DROP TABLE IF EXISTS crosswalk;")
        # stream in chunks for large CSVs
        total = 0
        for chunk in pd.read_csv(csv_path, dtype=str, chunksize=chunksize):
            chunk = chunk.rename(columns=str.strip).rename(columns=str.lower)
            # normalize expected columns names
            chunk = chunk.rename(columns={
                "tow": "tow_code",
                "towkod": "tow_code",
                "tow_kod": "tow_code",
                "supplier": "supplier_id",
                "vendor": "vendor_id",
                "vendor_navi": "vendor_id",
            })
            for c in ["tow_code","supplier_id","vendor_id"]:
                if c not in chunk.columns:
                    raise ValueError(f"CSV must contain column '{c}'")
            chunk = chunk[["tow_code","supplier_id","vendor_id"]].fillna("")
            chunk.to_sql("crosswalk", con, if_exists="append", index=False)
            total += len(chunk)
            print(f"• chunk: upserted {len(chunk)} rows (total {total})")

        # add the unique index for upsert
        con.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
            ON crosswalk(vendor_id, supplier_id);
        """)
    print(f"✓ Done. Total rows in DB: {total}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--db", default="data/crosswalk.db")
    args = ap.parse_args()
    rebuild_db(Path(args.csv), Path(args.db))
