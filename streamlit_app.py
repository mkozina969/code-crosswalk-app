import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from io import BytesIO

st.set_page_config(page_title="Supplier → TOW Mapper (SQLite)", layout="wide")

# -----------------------------------------------------------------------------
# 1) Crosswalk loader (robust to tow/tow_code and supplier_code/supplier_id)
# -----------------------------------------------------------------------------
DB_CANDIDATES = [Path("data/crosswalk.db"), Path("crosswalk.db")]

@st.cache_data(show_spinner=False)
def load_crosswalk_standardized() -> pd.DataFrame:
    """
    Open a crosswalk SQLite DB and return a DataFrame with canonical columns:
      - supplier_id (normalized from supplier_id/supplier_code)
      - tow         (normalized from tow/tow_code)
      - vendor_id   (optional)
    """
    for p in DB_CANDIDATES:
        if p.exists():
            con = sqlite3.connect(str(p))
            try:
                tables = con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                pragma = con.execute("PRAGMA table_info(crosswalk)").fetchall()

                # Show what we actually opened (helps debugging)
                st.caption(f"DB used: {p} | tables: {[t[0] for t in tables]}")
                st.caption(f"PRAGMA crosswalk: {[(c[1], c[2]) for c in pragma]}")

                # Map available columns by lower name
                cols_map = {c[1].lower(): c[1] for c in pragma}
                tow_col = cols_map.get("tow") or cols_map.get("tow_code")
                sup_col = cols_map.get("supplier_id") or cols_map.get("supplier_code")
                ven_col = cols_map.get("vendor_id")  # optional

                if not tow_col or not sup_col:
                    raise ValueError(
                        f"Crosswalk needs TOW and Supplier columns. "
                        f"Found: {list(cols_map.keys())}"
                    )

                sql = f'SELECT "{sup_col}" AS supplier_id, "{tow_col}" AS tow'
                if ven_col:
                    sql += f', "{ven_col}" AS vendor_id'
                sql += " FROM crosswalk"

                df = pd.read_sql_query(sql, con)
            finally:
                con.close()

            # Normalize
            df["supplier_id"] = df["supplier_id"].astype(str).str.strip().str.upper()
            df["tow"] = df["tow"].astype(str).str.strip()
            if "vendor_id" in df.columns:
                df["vendor_id"] = df["vendor_id"].astype(str).str.strip().str.upper()

            return df

    raise FileNotFoundError(
        "No crosswalk DB found. Expected at data/crosswalk.db or ./crosswalk.db"
    )

# Fallback to CSV when DB is missing (optional but handy)
@st.cache_data(show_spinner=False)
def load_crosswalk_from_csv() -> pd.DataFrame:
    candidates = [Path("data/crosswalk.csv"), Path("crosswalk.csv")]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, engine="python")
            cols = {c.lower(): c for c in df.columns}
            tow_col = cols.get("tow") or cols.get("tow_code")
            sup_col = cols.get("supplier_id") or cols.get("supplier_code")
            ven_col = cols.get("vendor_id")
            if not tow_col or not sup_col:
                raise ValueError(
                    f"CSV crosswalk must have tow/tow_code and supplier_id/supplier_code. "
                    f"Found: {list(df.columns)}"
                )
            out = pd.DataFrame({
                "supplier_id": df[sup_col].astype(str).str.strip().str.upper(),
                "tow": df[tow_col].astype(str).str.strip()
            })
            if ven_col:
                out["vendor_id"] = df[ven_col].astype(str).str.strip().str.upper()
            st.caption(f"Loaded crosswalk CSV: {p}")
            return out
    raise FileNotFoundError("No crosswalk CSV found (data/crosswalk.csv or crosswalk.csv).")


# -----------------------------------------------------------------------------
# 2) Help box
# -----------------------------------------------------------------------------
with st.expander("How to use", expanded=True):
    st.markdown("""
1. The app opens a **crosswalk SQLite DB** at `data/crosswalk.db` or `./crosswalk.db`.  
   If missing, it tries **crosswalk.csv** as a fallback.
2. Crosswalk must contain **tow** (or `tow_code`), **supplier_id** (or `supplier_code`), and optional **vendor_id**.
3. Upload a supplier **invoice** (Excel/CSV).
4. Pick the **Vendor** (or `ALL`) and choose the invoice **Supplier Code** column.
5. Download **Matched** and **Unmatched** Excel files.
""")

# -----------------------------------------------------------------------------
# 3) Load crosswalk (standardized columns)
# -----------------------------------------------------------------------------
try:
    cw = load_crosswalk_standardized()
except FileNotFoundError:
    cw = load_crosswalk_from_csv()

st.success(
    f"Crosswalk loaded | rows: {len(cw):,} | "
    f"vendors: {cw['vendor_id'].nunique() if 'vendor_id' in cw.columns else 'N/A'}"
)

# -----------------------------------------------------------------------------
# 4) Vendor selector (only when vendor_id exists)
# -----------------------------------------------------------------------------
vendor = "ALL"
if "vendor_id" in cw.columns:
    vendors = ["ALL"] + sorted(cw["vendor_id"].dropna().unique().tolist())
    vendor = st.selectbox("Vendor", vendors, index=0)
else:
    st.caption("No vendor_id in crosswalk → using ALL.")

# Filter crosswalk by vendor if chosen
cw_for_vendor = cw
if vendor != "ALL" and "vendor_id" in cw.columns:
    cw_for_vendor = cw[cw["vendor_id"] == vendor]

# -----------------------------------------------------------------------------
# 5) Upload invoice
# -----------------------------------------------------------------------------
st.header("2) Upload supplier invoice (Excel / CSV)")
invoice_file = st.file_uploader(
    "Drag & drop or Browse", type=["xlsx", "xls", "csv"], accept_multiple_files=False
)

invoice_df = None
if invoice_file:
    try:
        if invoice_file.name.lower().endswith((".xlsx", ".xls")):
            invoice_df = pd.read_excel(invoice_file)
        else:
            invoice_df = pd.read_csv(invoice_file, engine="python")
    except Exception as e:
        st.error(f"Failed to load invoice: {e}")
        invoice_df = None

    if invoice_df is not None:
        st.write("Preview:", invoice_df.head(10))
        st.caption(f"Rows: {len(invoice_df):,} | Columns: {list(invoice_df.columns)}")

# -----------------------------------------------------------------------------
# --- Admin panel: add / queue / apply mappings --------------------------------
import csv, sqlite3, os
from pathlib import Path

DB_PATH = Path("data/crosswalk.db")
PENDING_CSV = Path("data/updates.csv")

def _ensure_data_dir():
    PENDING_CSV.parent.mkdir(parents=True, exist_ok=True)

def _ensure_unique_index(con):
    cur = con.cursor()
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_crosswalk_vendor_supplier
        ON crosswalk(vendor_id, supplier_id)
    """)
    con.commit()

def upsert_mapping_sqlite(vendor_id: str, supplier_id: str, tow_code: str) -> None:
    """Insert or update a single mapping into the SQLite DB."""
    _ensure_data_dir()
    con = sqlite3.connect(DB_PATH)
    try:
        _ensure_unique_index(con)
        cur = con.cursor()
        cur.execute("""
        INSERT INTO crosswalk (tow_code, supplier_id, vendor_id)
        VALUES (?, ?, ?)
        ON CONFLICT(vendor_id, supplier_id)
        DO UPDATE SET tow_code = excluded.tow_code
        """, (str(tow_code).strip(), str(supplier_id).strip(), str(vendor_id).strip() or None))
        con.commit()
    finally:
        con.close()

def append_pending_csv(vendor_id: str, supplier_id: str, tow_code: str) -> None:
    """Queue a mapping into data/updates.csv (created if missing)."""
    _ensure_data_dir()
    write_header = not PENDING_CSV.exists()
    with PENDING_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tow_code", "supplier_id", "vendor_id"])
        if write_header:
            w.writeheader()
        w.writerow({
            "tow_code": str(tow_code).strip(),
            "supplier_id": str(supplier_id).strip(),
            "vendor_id": str(vendor_id).strip()
        })

def apply_pending_to_sqlite() -> int:
    """Read data/updates.csv and upsert all rows into the DB. Returns count."""
    if not PENDING_CSV.exists():
        return 0
    con = sqlite3.connect(DB_PATH)
    try:
        _ensure_unique_index(con)
        cur = con.cursor()
        with PENDING_CSV.open(newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            n = 0
            for r in rdr:
                tow = str(r["tow_code"]).strip()
                sup = str(r["supplier_id"]).strip()
                ven = str(r.get("vendor_id", "")).strip() or None
                cur.execute("""
                INSERT INTO crosswalk (tow_code, supplier_id, vendor_id)
                VALUES (?, ?, ?)
                ON CONFLICT(vendor_id, supplier_id)
                DO UPDATE SET tow_code = excluded.tow_code
                """, (tow, sup, ven))
                n += 1
        con.commit()
        return n
    finally:
        con.close()

def load_pending_df():
    if PENDING_CSV.exists():
        return pd.read_csv(PENDING_CSV, dtype=str)
    return pd.DataFrame(columns=["tow_code", "supplier_id", "vendor_id"])

# ------------- Admin UI -------------
with st.expander("🔐 Admin • Add / Queue / Apply Mappings", expanded=False):
    # Simple PIN gate (change this!). Preferably put PIN in st.secrets["admin_pin"]
    default_pin = os.environ.get("ST_ADMIN_PIN", None) or st.secrets.get("admin_pin", "letmein")
    pin = st.text_input("Admin PIN", type="password", placeholder="Enter PIN to enable admin actions")
    ok = st.button("Unlock")

    if ok:
        if pin != default_pin:
            st.error("Incorrect PIN.")
        else:
            st.success("Admin unlocked.")
            st.caption(f"DB: `{DB_PATH}`  •  Pending CSV: `{PENDING_CSV}`")

            st.subheader("Add a single mapping")
            with st.form("admin_add_one"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    vendor_id = st.text_input("vendor_id", placeholder="e.g. DOB0000025")
                with c2:
                    supplier_id = st.text_input("supplier_id", placeholder="e.g. 0986356023")
                with c3:
                    tow_code = st.text_input("tow_code", placeholder="e.g. 200183")

                add_mode = st.radio(
                    "Add to…",
                    ["Queue (updates.csv)", "Directly to SQLite (upsert)"],
                    horizontal=True,
                )
                submitted = st.form_submit_button("Add")
                if submitted:
                    if not (vendor_id and supplier_id and tow_code):
                        st.error("All three fields are required.")
                    else:
                        # Normalize as TEXT (keep leading zeros)
                        vendor_id = str(vendor_id).strip()
                        supplier_id = str(supplier_id).strip()
                        tow_code = str(tow_code).strip()

                        try:
                            if add_mode.startswith("Queue"):
                                append_pending_csv(vendor_id, supplier_id, tow_code)
                                st.success(f"Queued: {vendor_id} / {supplier_id} → {tow_code}")
                            else:
                                upsert_mapping_sqlite(vendor_id, supplier_id, tow_code)
                                st.success(f"Upserted into DB: {vendor_id} / {supplier_id} → {tow_code}")
                        except Exception as e:
                            st.exception(e)

            st.subheader("Queued updates (data/updates.csv)")
            df_pending = load_pending_df()
            st.dataframe(df_pending, use_container_width=True, height=220)

            cA, cB, cC = st.columns([1,1,1])
            with cA:
                if st.button("Apply queued to SQLite (Upsert)"):
                    try:
                        n = apply_pending_to_sqlite()
                        st.success(f"Applied {n} row(s) to DB.")
                    except Exception as e:
                        st.exception(e)

            with cB:
                if st.download_button(
                    "Download queued CSV",
                    data=df_pending.to_csv(index=False).encode("utf-8"),
                    file_name="updates.csv",
                    mime="text/csv",
                    disabled=df_pending.empty,
                ):
                    pass

            with cC:
                if st.button("Clear queued CSV", type="secondary", disabled=df_pending.empty):
                    try:
                        PENDING_CSV.unlink(missing_ok=True)
                        st.info("Cleared data/updates.csv.")
                    except Exception as e:
                        st.exception(e)

# --- Admin: Live search / inspect mappings ------------------------------------
def _list_vendors():
    if not DB_PATH.exists():
        return []
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        cur.execute("SELECT DISTINCT vendor_id FROM crosswalk ORDER BY vendor_id")
        vals = [r[0] for r in cur.fetchall() if r[0] is not None and str(r[0]).strip() != ""]
        return vals
    finally:
        con.close()

def _search_mappings(vendor_filter: str | None, supplier_q: str, exact: bool):
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["vendor_id", "supplier_id", "tow_code"])
    con = sqlite3.connect(DB_PATH)
    try:
        base = "SELECT vendor_id, supplier_id, tow_code FROM crosswalk"
        clauses = []
        params = []
        if vendor_filter and vendor_filter != "ALL":
            clauses.append("vendor_id = ?")
            params.append(vendor_filter)
        if supplier_q:
            if exact:
                clauses.append("supplier_id = ?")
                params.append(supplier_q)
            else:
                clauses.append("supplier_id LIKE ?")
                params.append(f"%{supplier_q}%")
        if clauses:
            base += " WHERE " + " AND ".join(clauses)
        base += " ORDER BY vendor_id, supplier_id LIMIT 500"
        return pd.read_sql_query(base, con, params=params, dtype=str)
    finally:
        con.close()

with st.expander("🔎 Admin • Live search / inspect crosswalk", expanded=False):
    if "admin_pin_ok" not in st.session_state:
        st.info("Unlock the Admin section first to use this panel.")
    else:
        # small state flag you can set in the admin unlock (below) if you want strict gating
        pass

    # Vendor picker
    vendors = _list_vendors()
    vendor_opt = ["ALL"] + vendors
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        pick_vendor = st.selectbox("Vendor", vendor_opt, index=0)
    with c2:
        supplier_q = st.text_input("supplier_id search", placeholder="exact or contains…")
    with c3:
        exact = st.checkbox("Exact", value=True)

    df_res = _search_mappings(pick_vendor, supplier_q.strip(), exact)
    st.caption(f"{len(df_res)} result(s) shown (max 500)")
    sel = st.dataframe(
        df_res,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=260,
    )

    # If user selects a row, prefill the add/upsert controls above via session_state
    # We’ll store into session keys the Admin form reads (vendor_id / supplier_id / tow_code)
    if sel and "selection" in sel and sel["selection"].get("rows"):
        idx = sel["selection"]["rows"][0]
        row = df_res.iloc[int(idx)]
        st.info(f"Selected: vendor={row['vendor_id']} • supplier={row['supplier_id']} • tow={row['tow_code']}")
        # Pre-fill the "Add a single mapping" form (if you want this, set these keys there)
        st.session_state.setdefault("prefill_vendor_id", row["vendor_id"])
        st.session_state.setdefault("prefill_supplier_id", row["supplier_id"])
        st.session_state.setdefault("prefill_tow_code", row["tow_code"])
        st.caption("Go to the **Add a single mapping** form above — fields are prefilled.")

    st.divider()
    st.subheader("Bulk queue: upload CSV (tow_code, supplier_id, vendor_id)")

    up_file = st.file_uploader("Upload CSV to queue", type=["csv"], accept_multiple_files=False, key="bulkq")
    if up_file is not None:
        try:
            up_df = pd.read_csv(up_file, dtype=str)
            required = {"tow_code", "supplier_id", "vendor_id"}
            if not required.issubset(set(map(str.lower, up_df.columns))):
                st.error("CSV must contain columns: tow_code, supplier_id, vendor_id")
            else:
                # normalize headers (case-insensitive)
                cols = {c.lower(): c for c in up_df.columns}
                q = up_df.rename(columns={cols["tow_code"]: "tow_code",
                                          cols["supplier_id"]: "supplier_id",
                                          cols["vendor_id"]: "vendor_id"})
                q = q[["tow_code", "supplier_id", "vendor_id"]].fillna("")
                cnt = 0
                for _, r in q.iterrows():
                    append_pending_csv(str(r["vendor_id"]), str(r["supplier_id"]), str(r["tow_code"]))
                    cnt += 1
                st.success(f"Queued {cnt} mapping(s) into updates.csv")
        except Exception as e:
            st.exception(e)


# 6) Column picker + mapping
# -----------------------------------------------------------------------------
st.header("3) Map to TOW")

def suggest_supplier_column(cols):
    low = [c.lower() for c in cols]
    candidates = [
        "supplier_id", "supplier", "supplier code", "suppliercode", "supplier_cod",
        "code", "ean", "sku", "article", "catalog", "catalogue", "šifra", "sifra"
    ]
    for i, c in enumerate(low):
        if any(tok in c for tok in candidates):
            return i
    return 0

if invoice_df is not None:
    idx = suggest_supplier_column(invoice_df.columns)
    code_col = st.selectbox(
        "Which column contains the SUPPLIER code?",
        options=list(invoice_df.columns), index=idx
    )

    do_map = st.button("Run mapping")
    if do_map:
        try:
            # Normalize join keys on both sides
            left = invoice_df.copy()
            left["_supplier_id_norm"] = left[code_col].astype(str).str.strip().str.upper()

            right = cw_for_vendor.copy()
            right["_supplier_id_norm"] = right["supplier_id"].astype(str).str.strip().str.upper()

            merged = left.merge(
                right[["_supplier_id_norm", "tow"]],
                on="_supplier_id_norm", how="left"
            ).drop(columns=["_supplier_id_norm"])

            matched = merged[merged["tow"].notna()].copy()
            unmatched = merged[merged["tow"].isna()].copy()

            st.success(
                f"Mapping complete → matched: {len(matched):,} | unmatched: {len(unmatched):,}"
            )

            with st.expander("Preview: Matched (first 200 rows)", expanded=False):
                st.dataframe(matched.head(200), use_container_width=True)

            with st.expander("Preview: Unmatched (first 200 rows)", expanded=False):
                st.dataframe(unmatched.head(200), use_container_width=True)

            # Excel download (two sheets)
            def to_excel_bytes(df_dict: dict) -> bytes:
                bio = BytesIO()
                with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
                    for sheet, df in df_dict.items():
                        df.to_excel(writer, index=False, sheet_name=sheet)
                return bio.getvalue()

            xls_bytes = to_excel_bytes({"Matched": matched, "Unmatched": unmatched})
            st.download_button(
                "Download Excel (Matched + Unmatched)",
                data=xls_bytes,
                file_name="mapping_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Mapping failed: {e}")
else:
    st.info("Upload your supplier invoice to enable mapping.")
