import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from io import BytesIO

st.set_page_config(page_title="Supplier → TOW Mapper (SQLite)", layout="wide")

# =============================================================================
# 1) Crosswalk loader (DB or CSV), standardize column names
# =============================================================================
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

# =============================================================================
# 2) Help box
# =============================================================================
with st.expander("How to use", expanded=True):
    st.markdown("""
1. The app opens a **crosswalk SQLite DB** at `data/crosswalk.db` or `./crosswalk.db`.  
   If missing, it tries **crosswalk.csv** as a fallback.
2. Crosswalk must contain **tow** (or `tow_code`), **supplier_id** (or `supplier_code`), and optional **vendor_id**.
3. Upload a supplier **invoice** (Excel/CSV).
4. Pick the **Vendor** (or `ALL`) and choose the invoice **Supplier Code** column.
5. Download **Matched** and **Unmatched** Excel files.
""")

# =============================================================================
# 3) Load crosswalk
# =============================================================================
try:
    cw = load_crosswalk_standardized()
except FileNotFoundError:
    cw = load_crosswalk_from_csv()

st.success(
    f"Crosswalk loaded | rows: {len(cw):,} | "
    f"vendors: {cw['vendor_id'].nunique() if 'vendor_id' in cw.columns else 'N/A'}"
)

# =============================================================================
# 4) Vendor selector (if vendor_id exists)
# =============================================================================
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

# =============================================================================
# 5) Upload invoice
# =============================================================================
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

# =============================================================================
# 6) Column picker + mapping (keeps your original stable merge)
# =============================================================================
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

# =============================================================================
# 7) Admin (optional) — UPSERT single mapping and live search
# =============================================================================
st.markdown("---")
with st.expander("🔐 Admin · Add / Live search / Apply mapping (optional)"):
    pin = st.text_input("Admin PIN", type="password", value="")
    if pin == "0000":
        st.success("Admin unlocked.")
        # DB path detection (same as loader order)
        db_path = next((str(p) for p in DB_CANDIDATES if p.exists()), None)
        st.caption(f"DB: {db_path or 'N/A'}")

        if not db_path:
            st.warning("No SQLite DB found on disk. Admin upserts require a DB (not CSV fallback).")
        else:
            # Check schema for vendor_id
            con = sqlite3.connect(db_path)
            try:
                pragma = con.execute("PRAGMA table_info(crosswalk)").fetchall()
                cols = {c[1].lower(): c[1] for c in pragma}
                has_vendor = "vendor_id" in cols

                # Ensure unique index used by ON CONFLICT exists
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
            finally:
                con.close()

            st.subheader("Add a single mapping (direct UPSERT to SQLite)")
            v_in = st.text_input("vendor_id", value=(vendor if vendor != "ALL" else ""))
            s_in = st.text_input("supplier_id")
            t_in = st.text_input("tow_code")

            if st.button("Add (Direct to SQLite)"):
                if not s_in or not t_in:
                    st.error("supplier_id and tow_code are required.")
                elif has_vendor and not v_in:
                    st.error("This crosswalk has vendor_id — you must provide vendor_id.")
                else:
                    con = sqlite3.connect(db_path)
                    try:
                        if has_vendor:
                            sql = """
                                INSERT INTO crosswalk(tow_code, supplier_id, vendor_id)
                                VALUES(?,?,?)
                                ON CONFLICT(vendor_id, supplier_id)
                                DO UPDATE SET tow_code = excluded.tow_code
                            """
                            con.execute(sql, (t_in.strip(), s_in.strip().upper(), v_in.strip().upper()))
                        else:
                            sql = """
                                INSERT INTO crosswalk(tow_code, supplier_id)
                                VALUES(?,?)
                                ON CONFLICT(supplier_id)
                                DO UPDATE SET tow_code = excluded.tow_code
                            """
                            con.execute(sql, (t_in.strip(), s_in.strip().upper()))
                        con.commit()
                        st.success("UPSERT complete.")
                    except Exception as e:
                        st.error(f"Insert failed: {e}")
                    finally:
                        con.close()

            st.subheader("Live search / inspect crosswalk")
            v_pick = st.selectbox(
                "Vendor", ["ALL"] + sorted(cw["vendor_id"].dropna().unique().tolist()) if "vendor_id" in cw.columns else ["ALL"],
                index=0
            )
            sup_like = st.text_input("supplier_id contains ...", value="")
            limit = st.slider("limit", 10, 500, 100, 10)

            # Query from disk so we see fresh UPSERTs
            if db_path:
                con = sqlite3.connect(db_path)
                try:
                    if "vendor_id" in cw.columns and v_pick != "ALL":
                        sql = """
                            SELECT tow_code AS tow, supplier_id, vendor_id
                            FROM crosswalk
                            WHERE vendor_id = ?
                              AND supplier_id LIKE ?
                            ORDER BY supplier_id
                            LIMIT ?
                        """
                        dfv = pd.read_sql_query(sql, con, params=[v_pick, f"%{sup_like.upper()}%", limit])
                    else:
                        sql = """
                            SELECT tow_code AS tow, supplier_id
                                 , CASE WHEN instr(lower(group_concat(name)), 'vendor_id') > 0
                                   THEN vendor_id ELSE NULL END AS vendor_id
                            FROM crosswalk
                            WHERE supplier_id LIKE ?
                            ORDER BY supplier_id
                            LIMIT ?
                        """
                        # That CASE is a harmless placeholder in case vendor_id doesn't exist.
                        # If vendor_id exists, SQLite will bind it; else it will return NULL.
                        try:
                            dfv = pd.read_sql_query(
                                "SELECT tow_code AS tow, supplier_id, vendor_id FROM crosswalk "
                                "WHERE supplier_id LIKE ? ORDER BY supplier_id LIMIT ?",
                                con, params=[f"%{sup_like.upper()}%", limit]
                            )
                        except Exception:
                            dfv = pd.read_sql_query(
                                "SELECT tow_code AS tow, supplier_id FROM crosswalk "
                                "WHERE supplier_id LIKE ? ORDER BY supplier_id LIMIT ?",
                                con, params=[f"%{sup_like.upper()}%", limit]
                            )

                    st.dataframe(dfv, use_container_width=True)
                finally:
                    con.close()
    else:
        st.info("Enter PIN to unlock Admin (default: 0000).")
