# streamlit_app.py
# -----------------------------------------------------------------------------
# Supplier -> TOW Mapper (SQLite first) with CSV fallback and Admin tools
# Repo is the source of truth: app opens data/crosswalk.db from the repo.
# If missing, it builds the DB from crosswalk.csv on the fly (read-only thereafter).
#
# Admin can either upsert directly into SQLite (if the environment is writable)
# or queue changes into data/updates.csv for a Git PR + DB rebuild.
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import os
from pathlib import Path
import sqlite3
from typing import Iterable, Optional, Tuple

import pandas as pd
import streamlit as st


# --------------------------- Repo layout constants ---------------------------
DB_PATH = Path("data/crosswalk.db")
CSV_PATH = Path("crosswalk.csv")           # the master CSV in repo
UPDATES_CSV = Path("data/updates.csv")     # queued admin updates (append only)

REPO_READONLY_MODE = True  # treat repo as source of truth; no forced writes
ADMIN_PIN = os.environ.get("ADMIN_PIN", "0000")

# --------------------------- Utilities: DB open/build ------------------------
def ensure_db_from_csv_if_missing() -> None:
    """
    If data/crosswalk.db doesn't exist but crosswalk.csv does, build it.
    This is safe in local dev. On Streamlit Cloud this will succeed
    in the ephemeral file system (not committed) but is fine for runtime.
    """
    if DB_PATH.exists():
        return

    if not CSV_PATH.exists():
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    df_iter = pd.read_csv(CSV_PATH, dtype=str, chunksize=100_000)
    first = True
    for chunk in df_iter:
        chunk = normalize_crosswalk_df(chunk)
        chunk.to_sql("crosswalk", con, if_exists="append" if not first else "replace", index=False)
        first = False

    # Ensure the unique index for ON CONFLICT
    con.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
        ON crosswalk(vendor_id, supplier_id)
        """
    )
    con.commit()
    con.close()


def open_db() -> sqlite3.Connection:
    """
    Open the repo crosswalk DB and guarantee the unique index exists.
    """
    ensure_db_from_csv_if_missing()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code    TEXT,
            supplier_id TEXT,
            vendor_id   TEXT
        )
        """
    )
    con.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
        ON crosswalk(vendor_id, supplier_id)
        """
    )
    con.commit()
    return con


def normalize_crosswalk_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize headers + types of the crosswalk CSV/DF.
    Expected columns: tow_code, supplier_id, vendor_id (case-insensitive allowed).
    """
    cols = {c.strip().lower(): c for c in df.columns}
    # remap if user used other labels
    def pick(colnames: Iterable[str], fallback: Optional[str] = None) -> Optional[str]:
        for c in colnames:
            if c in cols:
                return cols[c]
        return fallback

    col_tow  = pick(["tow", "tow_code", "towkod", "tow_kod", "towcode"])
    col_sup  = pick(["supplier_id", "supplier", "vendor code", "vendor_code", "vendor code "])
    col_ven  = pick(["vendor_id", "vendor navi", "vendor_navi", "venodr_id", "vendorid"])

    # try stricter names if above didn't match
    if col_sup is None and "supplier_id" in df.columns:
        col_sup = "supplier_id"
    if col_tow is None and "tow_code" in df.columns:
        col_tow = "tow_code"
    if col_ven is None and "vendor_id" in df.columns:
        col_ven = "vendor_id"

    # rename to clean schema
    rename = {}
    if col_tow  and col_tow  != "tow_code":    rename[col_tow]  = "tow_code"
    if col_sup  and col_sup  != "supplier_id": rename[col_sup]  = "supplier_id"
    if col_ven  and col_ven  != "vendor_id":   rename[col_ven]  = "vendor_id"
    if rename:
        df = df.rename(columns=rename)

    for c in ["tow_code", "supplier_id", "vendor_id"]:
        if c not in df.columns:
            df[c] = None

    # cast to str and strip
    for c in ["tow_code", "supplier_id", "vendor_id"]:
        df[c] = df[c].astype(str).str.strip()

    # keep only three columns
    return df[["tow_code", "supplier_id", "vendor_id"]]


def upsert_mapping(con: sqlite3.Connection, vendor_id: str, supplier_id: str, tow_code: str) -> None:
    """
    Upsert mapping for (vendor_id, supplier_id) -> tow_code with ON CONFLICT
    using the ux_crosswalk_vs unique index.
    """
    con.execute(
        """
        INSERT INTO crosswalk(tow_code, supplier_id, vendor_id)
        VALUES (?, ?, ?)
        ON CONFLICT(vendor_id, supplier_id)
        DO UPDATE SET tow_code = excluded.tow_code
        """,
        (tow_code, supplier_id, vendor_id),
    )
    con.commit()


def load_crosswalk_df(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT tow_code, supplier_id, vendor_id FROM crosswalk", con, dtype=str)
    for c in ["tow_code", "supplier_id", "vendor_id"]:
        df[c] = df[c].astype(str).str.strip()
    return df


# --------------------------- UI helpers --------------------------------------
def smart_detect_supplier_column(df: pd.DataFrame) -> str:
    candidates = [
        "supplier_id", "supplier", "vendor code", "customs code",
        "supplier code", "suppliercode"
    ]
    lowers = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lowers:
            return lowers[k]
    # default to first
    return df.columns[0]


def df_to_excel_bytes(sheets: list[Tuple[str, pd.DataFrame]]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, df in sheets:
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return bio.getvalue()


# =============================== APP =========================================
st.set_page_config(page_title="Supplier → TOW Mapper (SQLite)", layout="wide")

st.title("Supplier → TOW Mapper (SQLite)")
st.caption("Opens **data/crosswalk.db** from the repo. If missing, loads from **crosswalk.csv** and builds a temporary DB.")

# show banner: DB and row count
con = open_db()
with con:
    row_count = con.execute("SELECT COUNT(*) FROM crosswalk").fetchone()[0]
vendors = pd.read_sql_query("SELECT DISTINCT vendor_id FROM crosswalk ORDER BY vendor_id", con, dtype=str)["vendor_id"].tolist()
st.success(f"DB: `{DB_PATH.as_posix()}` • rows: **{row_count:,}** • vendors: **{len(vendors)}**")

# --------------------------- Vendor selection --------------------------------
vendor_opt = ["ALL"] + vendors
pick_vendor = st.selectbox("Vendor", vendor_opt, index=0, key="vendor_select")

# --------------------------- Upload supplier invoice --------------------------
st.header("2) Upload supplier invoice (Excel / CSV)")

invoice = st.file_uploader("Drag & drop or Browse", type=["xlsx", "xls", "csv"])
if invoice:
    # Read invoice
    if invoice.name.lower().endswith(".csv"):
        inv_df = pd.read_csv(invoice, dtype=str)
    else:
        inv_df = pd.read_excel(invoice, dtype=str)

    st.caption(f"Preview ({len(inv_df):,} rows) • Columns: {', '.join(list(inv_df.columns))[:120]}{'…' if len(inv_df.columns)>10 else ''}")
    st.dataframe(inv_df.head(10), use_container_width=True)

    sup_col = st.selectbox("Which column contains the SUPPLIER code?", options=inv_df.columns.tolist(),
                           index=inv_df.columns.get_loc(smart_detect_supplier_column(inv_df)))
    st.divider()

    # --------------------- 3) Map to TOW -------------------------------------
    st.header("3) Map to TOW")

    cw_df = load_crosswalk_df(con)
    if pick_vendor != "ALL":
        cw_df = cw_df.loc[cw_df["vendor_id"] == pick_vendor]

    left = inv_df[[sup_col]].rename(columns={sup_col: "supplier_id"})
    left["supplier_id"] = left["supplier_id"].astype(str).str.strip()
    joined = left.merge(cw_df, on="supplier_id", how="left")

    matched   = joined[joined["tow_code"].notna()].copy()
    unmatched = joined[joined["tow_code"].isna()].copy()

    st.success(f"Mapping complete → matched: **{len(matched):,}** | unmatched: **{len(unmatched):,}**")

    with st.expander("Preview: Matched (first 200 rows)", expanded=False):
        st.dataframe(matched.head(200), use_container_width=True)
    with st.expander("Preview: Unmatched (first 200 rows)", expanded=False):
        st.dataframe(unmatched.head(200), use_container_width=True)

    # download both in one excel
    xls = df_to_excel_bytes([("Matched", matched), ("Unmatched", unmatched)])
    st.download_button("Download Excel (Matched + Unmatched)", data=xls, file_name="mapping_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ============================== Admin tools ==================================
st.divider()
with st.expander("🔐 Admin • Add / Queue / Live search"):
    pin_input = st.text_input("Admin PIN", type="password", value="", help="Set env var ADMIN_PIN to change (default '0000').", key="pin")
    unlocked = pin_input == ADMIN_PIN

    if not unlocked:
        st.warning("Enter PIN to unlock.")
        st.stop()
    else:
        st.success("Admin unlocked.")

    # DB info line
    with con:
        rc = con.execute("SELECT COUNT(*) FROM crosswalk").fetchone()[0]
    st.caption(f"DB: `{DB_PATH.as_posix()}` • Current rows: **{rc:,}**")

    st.subheader("Add a single mapping (direct UPSERT to SQLite)")
    c1, c2, c3 = st.columns(3)
    with c1:
        a_vendor_id = st.text_input("vendor_id", value="", placeholder="e.g. DOB0000025")
    with c2:
        a_supplier_id = st.text_input("supplier_id", value="", placeholder="e.g. 0986356023")
    with c3:
        a_tow = st.text_input("tow_code", value="", placeholder="e.g. 200183")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Add (Direct to SQLite)"):
            try:
                if not (a_vendor_id and a_supplier_id and a_tow):
                    st.error("Please fill all three fields.")
                else:
                    upsert_mapping(con, a_vendor_id.strip(), a_supplier_id.strip(), a_tow.strip())
                    st.success(f"Upserted: ({a_vendor_id}, {a_supplier_id}) → {a_tow}")
            except Exception as e:
                st.error(f"Insert failed: {e}")

    with colB:
        if st.button("Queue to data/updates.csv"):
            try:
                UPDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
                row = pd.DataFrame(
                    [{"vendor_id": a_vendor_id.strip(), "supplier_id": a_supplier_id.strip(), "tow_code": a_tow.strip()}]
                )
                if UPDATES_CSV.exists():
                    row.to_csv(UPDATES_CSV, mode="a", header=False, index=False)
                else:
                    row.to_csv(UPDATES_CSV, index=False)
                st.success(f"Queued to {UPDATES_CSV.as_posix()} (commit & PR to make it permanent).")
            except Exception as e:
                st.error(f"Queue failed: {e}")

    # live search
    st.subheader("Live search / inspect crosswalk")
    v_pick = st.selectbox("Vendor", ["ALL"] + vendors, index=0, key="live_vendor")
    s_contains = st.text_input("supplier_id contains …", value="", key="live_sup")

    query = "SELECT tow_code, supplier_id, vendor_id FROM crosswalk"
    params = []
    where = []
    if v_pick != "ALL":
        where.append("vendor_id = ?")
        params.append(v_pick)
    if s_contains.strip():
        where.append("supplier_id LIKE ?")
        params.append(f"%{s_contains.strip()}%")
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY vendor_id, supplier_id LIMIT 500"

    live = pd.read_sql_query(query, con, params=params, dtype=str)
    st.caption(f"{len(live)} result(s) shown (max 500)")
    st.dataframe(live, use_container_width=True)
