import sqlite3
import pandas as pd
import argparse
from pathlib import Path

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c.lower().strip(): c for c in df.columns})
    df = df.rename(columns={
        "tow": "tow_code",
        "tow_code": "tow_code",
        "supplier_id": "supplier_id",
        "vendor_id": "vendor_id"
    })
    for col in ["tow_code", "supplier_id", "vendor_id"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(str).str.strip()
    return df[["tow_code", "supplier_id", "vendor_id"]]

def build(csv: Path, db: Path):
    con = sqlite3.connect(db)
    first = True
    for chunk in pd.read_csv(csv, dtype=str, sep=None, engine="python", chunksize=100_000):
        norm = normalize(chunk)
        norm.to_sql("crosswalk", con, if_exists="replace" if first else "append", index=False)
        first = False

    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
        ON crosswalk(vendor_id, supplier_id)
    """)
    con.commit()
    con.close()
    print(f"✓ Done: built {db} from {csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    args = ap.parse_args()
    build(args.csv, args.db)
