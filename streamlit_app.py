import io
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "Supplier → TOW Mapper (SQLite)"
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "crosswalk.db"
CSV_PATH = Path("crosswalk.csv")

ADMIN_PIN = os.getenv("CROSSWALK_ADMIN_PIN", "4321")  # change if you want

# ---------- SQLite helpers ----------

def get_conn():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    return sqlite3.connect(DB_PATH)

def ensure_schema():
    with closing(get_conn()) as con, con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS crosswalk(
                tow_code TEXT,
                supplier_id TEXT,
                vendor_id TEXT
            );
        """)
        # Unique index for UPSERT (vendor_id + supplier_id)
        con.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
            ON crosswalk(vendor_id, supplier_id);
        """)

def load_crosswalk_from_csv_if_needed():
    """Rebuild DB from CSV if DB missing and CSV exists."""
    if DB_PATH.exists():
        return
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
        with closing(get_conn()) as con, con:
            con.execute("DROP TABLE IF EXISTS crosswalk;")
            df.to_sql("crosswalk", con, index=False)
        ensure_schema()

def get_vendor_list():
    with closing(get_conn()) as con:
        vendors = pd.read_sql("SELECT DISTINCT vendor_id FROM crosswalk ORDER BY vendor_id", con)["vendor_id"].tolist()
    return ["ALL"] + vendors

def upsert_mapping(vendor_id: str, supplier_id: str, tow_code: str):
    sql = """
    INSERT INTO crosswalk(tow_code, supplier_id, vendor_id)
    VALUES(?,?,?)
    ON CONFLICT(vendor_id, supplier_id)
    DO UPDATE SET tow_code=excluded.tow_code;
    """
    with closing(get_conn()) as con, con:
        con.execute(sql, (tow_code, supplier_id, vendor_id))

def export_db_to_csv() -> bytes:
    with closing(get_conn()) as con:
        df = pd.read_sql("SELECT tow_code, supplier_id, vendor_id FROM crosswalk ORDER BY vendor_id, supplier_id", con)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

def live_search(vendor: str, supplier_like: str, limit: int = 500):
    wh = []
    params = []
    if vendor and vendor != "ALL":
        wh.append("vendor_id = ?")
        params.append(vendor)
    if supplier_like:
        wh.append("supplier_id LIKE ?")
        params.append(f"%{supplier_like}%")
    where = "WHERE " + " AND ".join(wh) if wh else ""
    sql = f"""
      SELECT tow_code, supplier_id, vendor_id
      FROM crosswalk
      {where}
      ORDER BY vendor_id, supplier_id
      LIMIT ?
    """
    params.append(limit)
    with closing(get_conn()) as con:
        return pd.read_sql(sql, con, params=params)

# ---------- Invoice upload / mapping ----------

def read_invoice(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded, dtype=str).fillna("")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded, dtype=str).fillna("")
    raise ValueError("Please upload CSV / XLSX")

def guess_supplier_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "supplier_id","supplier","supplier code","suppliercode",
        "customs code","custom code","customs_code","custom code","customs",
        "custom code","sup_code","sup code","supcode","code","šifra","sifra"
    ]
    cols_norm = {c.lower().strip(): c for c in df.columns}
    for key in candidates:
        if key in cols_norm:
            return cols_norm[key]
    # heuristics: any column that is mostly digits/letters and “code-ish”
    for c in df.columns:
        s = df[c].astype(str)
        if s.str.len().median() in (5,6,7,8,9,10,11):
            return c
    return None

def run_mapping(df: pd.DataFrame, supplier_col: str, pick_vendor: str):
    """Join invoice to crosswalk by supplier_id (and vendor if chosen)."""
    if supplier_col not in df.columns:
        raise ValueError("Selected supplier column not in invoice.")
    inv = df.copy()
    inv["supplier_id"] = inv[supplier_col].astype(str).str.strip()

    with closing(get_conn()) as con:
        cw = pd.read_sql("SELECT tow_code, supplier_id, vendor_id FROM crosswalk", con)

    if pick_vendor != "ALL":
        cw = cw[cw["vendor_id"] == pick_vendor]

    m = inv.merge(cw, on="supplier_id", how="left", validate="m:1")
    matched = m[~m["tow_code"].isna()].copy()
    unmatched = m[m["tow_code"].isna()].copy().drop(columns=["tow_code","vendor_id"], errors="ignore")
    return matched, unmatched

def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()

# ---------- UI ----------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# DB ready
ensure_schema()
load_crosswalk_from_csv_if_needed()

st.info(f"Database ready: `{DB_PATH}`")

with st.expander("How to use", expanded=False):
    st.markdown("""
1. Choose **Vendor** (or **ALL**).
2. Upload your supplier invoice (**CSV**/**Excel**).
3. Pick the **Supplier Code** column (auto-detected).
4. Click **Run mapping** to split **Matched** and **Unmatched**.
5. Download **Excel** outputs.
6. (Optional) Use **Admin** to add a missing mapping (UPSERT) or live-search the DB.
""")

# Vendor
vendor_opt = get_vendor_list()
pick_vendor = st.selectbox("Vendor", vendor_opt, index=0)

# Upload invoice
st.header("2) Upload supplier invoice (Excel / CSV)")
uploaded = st.file_uploader("Drag & drop or Browse", type=["csv","xlsx","xls"], label_visibility="collapsed")

preview_df = None
supplier_col = None
if uploaded:
    try:
        preview_df = read_invoice(uploaded)
        st.caption(f"Rows: {len(preview_df):,} | Columns: {', '.join(preview_df.columns.astype(str)[:8])}{'...' if len(preview_df.columns)>8 else ''}")
        st.dataframe(preview_df.head(10), use_container_width=True)

        supplier_guess = guess_supplier_col(preview_df)
        supplier_col = st.selectbox("Which column contains the SUPPLIER code?", options=list(preview_df.columns),
                                    index=(list(preview_df.columns).index(supplier_guess) if supplier_guess in preview_df.columns else 0))
    except Exception as e:
        st.error(f"Failed to read invoice: {e}")

# Mapping
st.header("3) Map to TOW")
run = st.button("Run mapping", type="primary", disabled=(preview_df is None))
if run and preview_df is not None:
    try:
        matched, unmatched = run_mapping(preview_df, supplier_col, pick_vendor)
        st.success(f"Mapping complete → **matched:** {len(matched):,} | **unmatched:** {len(unmatched):,}")

        with st.expander("Preview: Matched (first 200 rows)", expanded=False):
            st.dataframe(matched.head(200), use_container_width=True, height=320)
        with st.expander("Preview: Unmatched (first 200 rows)", expanded=False):
            st.dataframe(unmatched.head(200), use_container_width=True, height=320)

        xls = to_excel_bytes({"Matched": matched, "Unmatched": unmatched})
        st.download_button("Download Excel (Matched + Unmatched)", xls, file_name="mapped_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Mapping failed: {e}")

# ---------- Admin ----------
st.header("🔒 Admin • Add / Live search / Export")
pin_col, btn_col = st.columns([1,1])
with pin_col:
    pin = st.text_input("Admin PIN", type="password", value="", help="Default can be set with env CROSSWALK_ADMIN_PIN")
with btn_col:
    unlocked = st.button("Unlock")

if unlocked and pin == ADMIN_PIN:
    st.success("Admin unlocked.")

    st.subheader("Add a single mapping (direct UPSERT to SQLite)")
    c1, c2, c3, c4 = st.columns([1,2,2,2])
    with c1:
        pass
    with c2:
        a_vendor = st.text_input("vendor_id", placeholder="e.g. DOB0000025")
    with c3:
        a_supplier = st.text_input("supplier_id", placeholder="e.g. 0986356023")
    with c4:
        a_tow = st.text_input("tow_code", placeholder="e.g. 200183")

    if st.button("Add (Direct to SQLite)"):
        try:
            if not (a_vendor and a_supplier and a_tow):
                raise ValueError("All three fields are required.")
            upsert_mapping(a_vendor.strip(), a_supplier.strip(), a_tow.strip())
            st.success("UPSERT OK.")
        except sqlite3.IntegrityError as ie:
            st.error(f"Insert failed (constraint): {ie}")
        except Exception as e:
            st.error(f"Insert failed: {e}")

    st.subheader("Live search / inspect crosswalk")
    vcol, scol = st.columns([1,2])
    with vcol:
        v_pick = st.selectbox("Vendor", get_vendor_list(), index=0, key="adm_vendor")
    with scol:
        s_like = st.text_input("supplier_id contains ...", key="adm_suplike")
    res = live_search(v_pick, s_like, limit=500)
    st.caption(f"{len(res)} result(s) shown (max 500)")
    st.dataframe(res, use_container_width=True, height=380)

    st.subheader("Export DB → CSV (for Git source of truth)")
    csv_bytes = export_db_to_csv()
    st.download_button("Download crosswalk.csv", data=csv_bytes, file_name="crosswalk.csv", mime="text/csv")

elif unlocked and pin != ADMIN_PIN:
    st.error("Wrong PIN.")
