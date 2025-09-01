import sqlite3
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "crosswalk.csv"
DB_PATH = BASE_DIR / "data" / "crosswalk.db"

def ensure_db():
    if DB_PATH.exists():
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    import make_crosswalk_db
    make_crosswalk_db.build(CSV_PATH, DB_PATH)

def open_db():
    ensure_db()
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code TEXT,
            supplier_id TEXT,
            vendor_id TEXT
        )
    """)
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
        ON crosswalk(vendor_id, supplier_id)
    """)
    con.commit()
    return con

st.title("Crosswalk Mapper")

con = open_db()

st.write("Database ready:", DB_PATH)

supplier_id = st.text_input("Supplier ID")
vendor_id = st.text_input("Vendor ID")
tow_code = st.text_input("TOW Code")

if st.button("Add mapping"):
    con.execute("""
        INSERT INTO crosswalk(tow_code, supplier_id, vendor_id)
        VALUES(?,?,?)
        ON CONFLICT(vendor_id, supplier_id)
        DO UPDATE SET tow_code=excluded.tow_code
    """, (tow_code, supplier_id, vendor_id))
    con.commit()
    st.success("Mapping saved")

if st.button("Show sample"):
    df = pd.read_sql("SELECT * FROM crosswalk LIMIT 20", con)
    st.dataframe(df)
