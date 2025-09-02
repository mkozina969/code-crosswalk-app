# streamlit_app.py
from __future__ import annotations

import io
import os
import sqlite3
from contextlib import closing
from typing import List, Tuple

import pandas as pd
import streamlit as st


# =========================
# Config
# =========================
DB_PATH = "data/crosswalk.db"
UPDATES_CSV = "data/updates.csv"  # queued upserts (optional)
ADMIN_PIN = "4321"                # change if you want


# =========================
# DB Helpers
# =========================
def get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path, isolation_level=None)  # autocommit
    con.execute("PRAGMA foreign_keys=ON;")
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def ensure_schema() -> None:
    with closing(get_conn()) as con:
        # table
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS crosswalk (
                tow_code    TEXT,
                supplier_id TEXT,
                vendor_id   TEXT
            );
            """
        )
        # unique index for ON CONFLICT
        con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
            ON crosswalk(vendor_id, supplier_id);
            """
        )


def upsert_mapping(vendor_id: str, supplier_id: str, tow_code: str) -> None:
    """Single-row upsert used by the Admin 'Add (Direct to SQLite)' button."""
    vendor_id = (vendor_id or "").strip()
    supplier_id = (supplier_id or "").strip()
    tow_code = (tow_code or "").strip()
    if not (vendor_id and supplier_id and tow_code):
        raise ValueError("vendor_id, supplier_id and tow_code are required")

    with closing(get_conn()) as con:
        con.execute(
            """
            INSERT INTO crosswalk (vendor_id, supplier_id, tow_code)
            VALUES (?, ?, ?)
            ON CONFLICT(vendor_id, supplier_id)
            DO UPDATE SET tow_code = excluded.tow_code;
            """,
            (vendor_id, supplier_id, tow_code),
        )


def apply_updates_csv(csv_path: str = UPDATES_CSV) -> int:
    """
    Apply queued CSV upserts (columns: vendor_id, supplier_id, tow_code).
    Returns number of rows applied.
    """
    if not os.path.exists(csv_path):
        return 0

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if df.empty:
        return 0

    with closing(get_conn()) as con:
        cur = con.cursor()
        for row in df.itertuples(index=False):
            cur.execute(
                """
                INSERT INTO crosswalk (vendor_id, supplier_id, tow_code)
                VALUES (?, ?, ?)
                ON CONFLICT(vendor_id, supplier_id)
                DO UPDATE SET tow_code = excluded.tow_code;
                """,
                (row.vendor_id.strip(), row.supplier_id.strip(), row.tow_code.strip()),
            )
        con.commit()
    return len(df)


def get_distinct_vendors() -> List[str]:
    with closing(get_conn()) as con:
        rows = con.execute("SELECT DISTINCT vendor_id FROM crosswalk ORDER BY 1").fetchall()
    return [r[0] for r in rows if r[0]]


def search_crosswalk(
    vendor_id: str | None, supplier_like: str | None, limit: int = 200
) -> pd.DataFrame:
    q = "SELECT tow_code, supplier_id, vendor_id FROM crosswalk"
    params: List[str] = []
    where = []

    if vendor_id and vendor_id.upper() != "ALL":
        where.append("vendor_id = ?")
        params.append(vendor_id)

    if supplier_like:
        where.append("supplier_id LIKE ?")
        params.append(f"%{supplier_like}%")

    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY vendor_id, supplier_id LIMIT ?"
    params.append(limit)

    with closing(get_conn()) as con:
        df = pd.read_sql_query(q, con, params=params)
    return df


# =========================
# UI Helpers
# =========================
@st.cache_data(show_spinner=False)
def cached_vendors() -> List[str]:
    vs = get_distinct_vendors()
    return ["ALL"] + vs


def read_any_table(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, dtype=str)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file, dtype=str)
    raise ValueError("Unsupported file type (use CSV/XLSX/XLS)")


# =========================
# App
# =========================
st.set_page_config(page_title="Supplier → TOW Mapper (SQLite)", layout="wide")
st.title("Supplier → TOW Mapper (SQLite)")

# Boot
ensure_schema()
with closing(get_conn()) as con:
    row_count = con.execute("SELECT COUNT(*) FROM crosswalk").fetchone()[0]
st.success(f"Database ready at **{DB_PATH}** • rows: **{row_count:,}**")


# ============ 2) Upload supplier invoice
st.header("2) Upload supplier invoice (Excel / CSV)")
up = st.file_uploader("Drag & drop or Browse", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

invoice_df: pd.DataFrame | None = None
supplier_col = None

if up is not None:
    try:
        invoice_df = read_any_table(up).fillna("")
        st.caption(f"Preview: **{up.name}**  • rows: **{len(invoice_df):,}**")
        st.dataframe(invoice_df.head(10), use_container_width=True)

        # pick supplier column
        text_cols = [c for c in invoice_df.columns if invoice_df[c].dtype == "object"]
        supplier_col = st.selectbox(
            "Which column contains the SUPPLIER code?",
            options=text_cols,
            index=min(text_cols.index("supplier_id") if "supplier_id" in text_cols else 0, len(text_cols) - 1),
        )
    except Exception as e:
        st.error(f"Failed to read file: {e}")


# ============ 3) Map to TOW
st.header("3) Map to TOW")

col_left, _ = st.columns([1, 3])
with col_left:
    vendor_pick = st.selectbox("Vendor", cached_vendors(), index=0)

st.divider()

with st.expander("🔐 Admin • Add / Queue / Apply Mappings", expanded=False):
    pin = st.text_input("Admin PIN", type="password")
    unlocked = pin == ADMIN_PIN

    if unlocked:
        st.success("Admin unlocked.")
        st.caption(f"DB: **{DB_PATH}** • Pending CSV: **{UPDATES_CSV}**")

        c1, c2, c3 = st.columns(3)
        with c1:
            v_in = st.text_input("vendor_id", placeholder="e.g. DOB0000025")
        with c2:
            s_in = st.text_input("supplier_id", placeholder="e.g. 0986356023")
        with c3:
            t_in = st.text_input("tow_code", placeholder="e.g. 200183")

        add_mode = st.radio(
            "Add to…",
            options=["Queue (updates.csv)", "Directly to SQLite (upsert)"],
            horizontal=True,
            index=1,
            label_visibility="visible",
        )

        if st.button("Add mapping"):
            try:
                if add_mode.startswith("Direct"):
                    upsert_mapping(v_in, s_in, t_in)
                    st.success("Upserted into SQLite.")
                else:
                    os.makedirs(os.path.dirname(UPDATES_CSV), exist_ok=True)
                    queued = pd.DataFrame([{"vendor_id": v_in, "supplier_id": s_in, "tow_code": t_in}])
                    if os.path.exists(UPDATES_CSV):
                        old = pd.read_csv(UPDATES_CSV, dtype=str)
                        queued = pd.concat([old, queued], ignore_index=True)
                    queued.to_csv(UPDATES_CSV, index=False)
                    st.success(f"Queued to {UPDATES_CSV} (rows now: {len(queued):,}).")
            except Exception as e:
                st.error(f"Add failed: {e}")

        # Show queued CSV
        st.subheader("Queued updates (data/updates.csv)")
        if os.path.exists(UPDATES_CSV):
            qdf = pd.read_csv(UPDATES_CSV, dtype=str).fillna("")
            st.dataframe(qdf, use_container_width=True, height=220)
            cqa, cqb = st.columns([1, 1])
            with cqa:
                if st.button("Apply queued to SQLite (Upsert)"):
                    try:
                        n = apply_updates_csv()
                        st.success(f"Applied {n:,} upserts from queue.")
                    except Exception as e:
                        st.error(f"Apply failed: {e}")
            with cqb:
                if st.button("Clear queued CSV"):
                    try:
                        os.remove(UPDATES_CSV)
                        st.success("Queue cleared.")
                    except Exception as e:
                        st.error(f"Clear failed: {e}")
        else:
            st.info("Queue is empty.")
    else:
        st.info("Enter the admin PIN to unlock this block.")


st.subheader("Live search / inspect crosswalk")
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    vendor_search = st.selectbox("Vendor", cached_vendors(), index=0, key="live_vendor")
with c2:
    supplier_like = st.text_input("supplier_id contains …", key="live_supplier_like")
with c3:
    limit = st.number_input("rows (max 500)", min_value=1, max_value=500, value=200, step=10)

try:
    results = search_crosswalk(vendor_search, supplier_like, limit=int(limit))
    st.dataframe(results, use_container_width=True, height=320)
except Exception as e:
    st.error(f"Search failed: {e}")


# =========================
# Optional: simple invoice → mapping demo
# =========================
st.divider()
if invoice_df is not None and supplier_col:
    st.subheader("Try mapping your uploaded invoice")
    run = st.button("Run mapping preview")
    if run:
        with closing(get_conn()) as con:
            cross = pd.read_sql_query(
                "SELECT vendor_id, supplier_id, tow_code FROM crosswalk", con
            )
        df = invoice_df.copy()
        # You may need to set the vendor to a chosen value per-file; for demo we don't filter
        df = df.merge(
            cross,
            left_on=[supplier_col],
            right_on=["supplier_id"],
            how="left",
        )
        matched = df[df["tow_code"].notna()]
        unmatched = df[df["tow_code"].isna()]

        st.success(f"Mapping complete → matched: **{len(matched):,}** • unmatched: **{len(unmatched):,}**")

        with st.expander("Preview: Matched (first 200 rows)", expanded=False):
            st.dataframe(matched.head(200), use_container_width=True, height=260)

        with st.expander("Preview: Unmatched (first 200 rows)", expanded=False):
            st.dataframe(unmatched.head(200), use_container_width=True, height=260)

        # Export button
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
            matched.to_excel(xw, sheet_name="matched", index=False)
            unmatched.to_excel(xw, sheet_name="unmatched", index=False)
        st.download_button(
            "Download Excel (Matched + Unmatched)",
            data=out.getvalue(),
            file_name="mapping_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
