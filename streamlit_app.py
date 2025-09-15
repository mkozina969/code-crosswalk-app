# streamlit_app.py
from __future__ import annotations

import os
import csv
import hashlib
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
import streamlit as st
from urllib.request import urlopen

st.set_page_config(page_title="Supplier → TOW Mapper (SQLite)", layout="wide")

# =============================================================================
# Paths & constants
# =============================================================================
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "crosswalk.db"
CSV_FALLBACK_1 = DATA_DIR / "crosswalk.csv"
CSV_FALLBACK_2 = Path("crosswalk.csv")
PENDING_CSV = DATA_DIR / "updates.csv"
DB_CANDIDATES = [DB_PATH, Path("crosswalk.db")]  # prefer data/crosswalk.db

DEBUG = False


def _log(msg: str):
    if DEBUG:
        st.caption(f"🔎 {msg}")


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================================
# Cloud bootstrap: fetch DB from a URL on startup (so it persists across
# reboots)
# ==========================================================================


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def maybe_fetch_db_from_url(show_messages: bool = True) -> Optional[str]:
    """
    If st.secrets["DB_URL"] is set, download that .db on startup (or on
    demand) and place it at data/crosswalk.db. Optionally verify with
    st.secrets["DB_SHA256"]. Returns a status string or None.
    """
    db_url = st.secrets.get("DB_URL", "").strip()
    if not db_url:
        return None

    expected_sha = st.secrets.get("DB_SHA256", "").strip() or None
    _ensure_data_dir()

    try:
        if show_messages:
            st.info("Fetching crosswalk DB from remote URL…")
        with urlopen(db_url) as resp:
            content = resp.read()

        if expected_sha:
            actual_sha = sha256_bytes(content)
            if actual_sha.lower() != expected_sha.lower():
                raise RuntimeError(
                    f"SHA256 mismatch: expected {expected_sha}, "
                    f"got {actual_sha}"
                )

        DB_PATH.write_bytes(content)
        return (
            f"Downloaded DB to {DB_PATH} ({len(content)/1_048_576:.2f} MB)"
            f"{' • SHA OK' if expected_sha else ''}"
        )
    except Exception as e:
        if show_messages:
            st.error(f"DB download failed: {e}")
        return f"DB download failed: {e}"


# Automatically try to fetch on startup if URL present and DB missing
if not DB_PATH.exists() and st.secrets.get("DB_URL"):
    msg = maybe_fetch_db_from_url(show_messages=False)
    if msg and DEBUG:
        _log(msg)


# =======================================================================
# DB Stats (path, mtime, size, live COUNT)
# =======================================================================
st.caption(f"📦 DB path: {DB_PATH.resolve()}")
if DB_PATH.exists():
    st.caption(
        f"🕒 DB modified: "
        f"{datetime.fromtimestamp(DB_PATH.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
        f" • size: {DB_PATH.stat().st_size/1_048_576:.2f} MB"
    )
    if st.button("Count rows (live from SQLite)"):
        con = sqlite3.connect(DB_PATH)
        try:
            cnt = con.execute("SELECT COUNT(*) FROM crosswalk").fetchone()[0]
            st.success(f"SQLite says: {cnt:, } rows")
        finally:
            con.close()
else:
    st.warning(
        "DB not found at expected path; the app may fall back to CSV "
        "(if present)."
    )


# =======================================================================
# Crosswalk loaders (DB preferred, CSV fallback)
# =======================================================================


def _detect_crosswalk_columns(con: sqlite3.Connection) -> dict:
    pragma = con.execute("PRAGMA table_info(crosswalk)").fetchall()
    cols = {c[1].lower(): c[1] for c in pragma}
    tow_col = cols.get("tow") or cols.get("tow_code")
    supplier_col = cols.get("supplier_id") or cols.get("supplier_code")
    vendor_col = cols.get("vendor_id")
    if not tow_col or not supplier_col:
        raise ValueError(
            "Crosswalk must have tow/tow_code AND supplier_id/supplier_code. "
            f"Found: {list(cols.keys())}"
        )
    return {"supplier": supplier_col, "tow": tow_col, "vendor": vendor_col}


@st.cache_data(show_spinner=False)
def load_crosswalk_standardized() -> pd.DataFrame:
    for p in DB_CANDIDATES:
        if p.exists():
            con = sqlite3.connect(str(p))
            try:
                cols = _detect_crosswalk_columns(con)
                supplier, tow = cols["supplier"], cols["tow"]
                vendor = cols["vendor"]
                sql = f'SELECT "{supplier}" AS supplier_id, "{tow}" AS tow'
                if vendor:
                    sql += f', "{vendor}" AS vendor_id'
                sql += " FROM crosswalk"
                df = pd.read_sql_query(sql, con)
                _log(f"Loaded SQLite: {p}")
            finally:
                con.close()
            df["supplier_id"] = df["supplier_id"].astype(str).str.strip().str.upper()
            df["tow"] = df["tow"].astype(str).str.strip()
            if "vendor_id" in df.columns:
                df["vendor_id"] = df["vendor_id"].astype(str).str.strip().str.upper()
            return df
    raise FileNotFoundError("No crosswalk DB found (data/crosswalk.db or ./crosswalk.db).")

@st.cache_data(show_spinner=False)
def load_crosswalk_from_csv() -> pd.DataFrame:
    def read_any_csv(path: Path) -> pd.DataFrame:
        for enc in ("utf-8", "latin-1"):
            try:
                sample = path.read_bytes()[:2048].decode(enc, errors="ignore")
                first = (sample.splitlines() or [""])[0].lower()
                sep = ";" if first.count(";") > first.count(", ") else ", "
                skiprows = 1 if first.startswith("sep=") else 0
                return pd.read_csv(
                    path, sep=sep, engine="python", dtype=str,
                    encoding=enc, on_bad_lines="skip", skiprows=skiprows
                )
            except Exception:
                continue
        raise ValueError(f"Could not read CSV: {path}")

    for p in (CSV_FALLBACK_1, CSV_FALLBACK_2):
        if p.exists():
            df = read_any_csv(p)
            cols = {c.lower().strip(): c for c in df.columns}
            tow_col = cols.get("tow") or cols.get("tow_code")
            sup_col = cols.get("supplier_id") or cols.get("supplier_code")
            ven_col = cols.get("vendor_id")
            if not tow_col or not sup_col:
                raise ValueError(
                    f"CSV must have tow/tow_code and supplier_id/supplier_code. Found: {list(df.columns)}"
                )
            out = pd.DataFrame({
                "supplier_id": df[sup_col].astype(str).str.strip().str.upper(),
                "tow": df[tow_col].astype(str).str.strip()
            })
            if ven_col:
                out["vendor_id"] = df[ven_col].astype(str).str.strip().str.upper()
            _log(f"Loaded crosswalk CSV: {p}")
            return out
    raise FileNotFoundError("No crosswalk CSV found (data/crosswalk.csv or crosswalk.csv).")

# =============================================================================
# UI: How to use + cache controls + fetch-from-URL control
# =============================================================================
with st.expander("How to use", expanded=True):
    st.markdown("""
1) The app opens **data/crosswalk.db** (or `./crosswalk.db`). If missing, it tries **crosswalk.csv**.
2) Crosswalk needs **tow/tow_code**, **supplier_id/supplier_code**, optional **vendor_id**.
3) Upload an invoice (Excel/CSV), pick **Vendor** & **Supplier Code** column, then **Run mapping**.
4) Download **Matched**/**Unmatched** Excel.
5) Use **Admin** for upserts, queue, live search — and now **DB replace** (session) or **fetch DB from URL** (persistent).
""")

c0, c1, c2 = st.columns([1, 1, 3])
with c0:
    if st.button("♻️ Clear cache & re-run", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with c1:
    if st.button("🌐 Fetch DB from URL (now)", use_container_width=True):
        msg = maybe_fetch_db_from_url(show_messages=True)
        if msg:
            st.info(msg)
            st.cache_data.clear()
            st.rerun()

# =============================================================================
# Load crosswalk (DB preferred, else CSV)
# =============================================================================
try:
    cw = load_crosswalk_standardized()
    source = "SQLite"
except FileNotFoundError:
    cw = load_crosswalk_from_csv()
    source = "CSV"

st.success(
    f"Crosswalk loaded from **{source}** | rows: {len(cw):, } | "
    f"vendors: {cw['vendor_id'].nunique() if 'vendor_id' in cw.columns else 'N/A'}"
)

# =============================================================================
# Vendor selector
# =============================================================================
vendor = "ALL"
if "vendor_id" in cw.columns:
    vendors = ["ALL"] + sorted(cw["vendor_id"].dropna().unique().tolist())
    vendor = st.selectbox("Vendor", vendors, index=0)
else:
    st.caption("No vendor_id in crosswalk → using ALL.")

cw_for_vendor = cw if vendor == "ALL" or "vendor_id" not in cw.columns else cw[cw["vendor_id"] == vendor]

# =============================================================================
# Upload invoice
# =============================================================================
st.header("2) Upload supplier invoice (Excel / CSV)")
invoice_file = st.file_uploader(
    "Drag & drop or Browse", type=["xlsx", "xls", "csv"], accept_multiple_files=False
)

invoice_df = None
if invoice_file:
    try:
        if invoice_file.name.lower().endswith((".xlsx", ".xls")):
            invoice_df = pd.read_excel(invoice_file, dtype=str)  # dtype=str preserves leading zeros
        else:
            invoice_df = pd.read_csv(invoice_file, engine="python", dtype=str, encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Failed to load invoice: {e}")
        invoice_df = None

    if invoice_df is not None:
        st.write("Preview:", invoice_df.head(10))
        st.caption(f"Rows: {len(invoice_df):, } | Columns: {list(invoice_df.columns)}")

# =============================================================================
# Mapping
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

    if st.button("Run mapping"):
        try:
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

            st.success(f"Mapping complete → matched: {len(matched):, } | unmatched: {len(unmatched):, }")

            with st.expander("Preview: Matched (first 200 rows)", expanded=False):
                st.dataframe(matched.head(200), use_container_width=True)

            with st.expander("Preview: Unmatched (first 200 rows)", expanded=False):
                st.dataframe(unmatched.head(200), use_container_width=True)

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
# Admin helpers (DB upsert + queue)
# =============================================================================
def _ensure_unique_index(con: sqlite3.Connection, supplier_col: str, vendor_col: Optional[str]):
    if vendor_col:
        con.execute(
            f'CREATE UNIQUE INDEX IF NOT EXISTS ix_crosswalk_vendor_supplier '
            f'ON crosswalk("{vendor_col}", "{supplier_col}")'
        )
    else:
        con.execute(
            f'CREATE UNIQUE INDEX IF NOT EXISTS ix_crosswalk_supplier '
            f'ON crosswalk("{supplier_col}")'
        )
    con.commit()

def _detect_cols(con: sqlite3.Connection):
    cols = _detect_crosswalk_columns(con)
    return cols["supplier"], cols["tow"], cols["vendor"]

def upsert_mapping_sqlite(vendor_id: str, supplier_id: str, tow_code: str) -> None:
    _ensure_data_dir()
    con = sqlite3.connect(DB_PATH)
    try:
        supplier_col, tow_col, vendor_col = _detect_cols(con)
        _ensure_unique_index(con, supplier_col, vendor_col)
        if vendor_col:
            sql = (
                f'INSERT INTO crosswalk("{tow_col}", "{supplier_col}", "{vendor_col}") VALUES (?, ?, ?) '
                f'ON CONFLICT("{vendor_col}", "{supplier_col}") DO UPDATE SET "{tow_col}" = excluded."{tow_col}"'
            )
            params = (str(tow_code).strip(), str(supplier_id).strip(), str(vendor_id).strip())
        else:
            sql = (
                f'INSERT INTO crosswalk("{tow_col}", "{supplier_col}") VALUES (?, ?) '
                f'ON CONFLICT("{supplier_col}") DO UPDATE SET "{tow_col}" = excluded."{tow_col}"'
            )
            params = (str(tow_code).strip(), str(supplier_id).strip())
        con.execute(sql, params)
        con.commit()
    finally:
        con.close()

def append_pending_csv(vendor_id: str, supplier_id: str, tow_code: str) -> None:
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
    if not PENDING_CSV.exists():
        return 0
    con = sqlite3.connect(DB_PATH)
    try:
        supplier_col, tow_col, vendor_col = _detect_cols(con)
        _ensure_unique_index(con, supplier_col, vendor_col)
        n = 0
        with PENDING_CSV.open(newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                tow = str(r["tow_code"]).strip()
                sup = str(r["supplier_id"]).strip()
                ven = str(r.get("vendor_id", "")).strip()
                if vendor_col:
                    sql = (
                        f'INSERT INTO crosswalk("{tow_col}", "{supplier_col}", "{vendor_col}") VALUES (?, ?, ?) '
                        f'ON CONFLICT("{vendor_col}", "{supplier_col}") '
                        f'DO UPDATE SET "{tow_col}" = excluded."{tow_col}"'
                    )
                    params = (tow, sup, ven)
                else:
                    sql = (
                        f'INSERT INTO crosswalk("{tow_col}", "{supplier_col}") VALUES (?, ?) '
                        f'ON CONFLICT("{supplier_col}") DO UPDATE SET "{tow_col}" = excluded."{tow_col}"'
                    )
                    params = (tow, sup)
                con.execute(sql, params)
                n += 1
        con.commit()
        return n
    finally:
        con.close()

def load_pending_df():
    if PENDING_CSV.exists():
        return pd.read_csv(PENDING_CSV, dtype=str)
    return pd.DataFrame(columns=["tow_code", "supplier_id", "vendor_id"])

# =============================================================================
# Admin panel (PIN-gated) + DB maintenance (upload replace)
# =============================================================================
with st.expander("🔐 Admin • Add / Queue / Apply Mappings", expanded=False):
    # PIN: env > secrets > fallback
    default_pin = os.environ.get("ST_ADMIN_PIN") or st.secrets.get("admin_pin", "letmein")
    col_pin, col_btn = st.columns([3, 1])
    with col_pin:
        pin = st.text_input("Admin PIN", type="password", placeholder="Enter PIN to enable admin actions")
    with col_btn:
        ok = st.button("Unlock", use_container_width=True)

    if ok:
        if pin != default_pin:
            st.error("Incorrect PIN.")
        else:
            st.success("Admin unlocked.")
            st.session_state["admin_pin_ok"] = True
            st.caption(f"DB: `{DB_PATH}`  •  Pending CSV: `{PENDING_CSV}`")

    if st.session_state.get("admin_pin_ok"):
        st.subheader("Add a single mapping")
        with st.form("admin_add_one"):
            c1, c2, c3 = st.columns(3)
            with c1:
                vendor_id_in = st.text_input("vendor_id",
                                             value=st.session_state.pop("prefill_vendor_id", ""),
                                             placeholder="e.g. DOB0000025")
            with c2:
                supplier_id_in = st.text_input("supplier_id",
                                               value=st.session_state.pop("prefill_supplier_id", ""),
                                               placeholder="e.g. 0986356023")
            with c3:
                tow_code_in = st.text_input("tow_code",
                                            value=st.session_state.pop("prefill_tow_code", ""),
                                            placeholder="e.g. 200183")

            add_mode = st.radio("Add to…", ["Queue (updates.csv)", "Directly to SQLite (upsert)"], horizontal=True)
            submitted = st.form_submit_button("Add")
            if submitted:
                if not (supplier_id_in and tow_code_in):
                    st.error("supplier_id and tow_code are required (vendor_id optional if DB has no vendor).")
                else:
                    try:
                        if add_mode.startswith("Queue"):
                            append_pending_csv(vendor_id_in, supplier_id_in, tow_code_in)
                            st.success(f"Queued: {vendor_id_in} / {supplier_id_in} → {tow_code_in}")
                        else:
                            upsert_mapping_sqlite(vendor_id_in, supplier_id_in, tow_code_in)
                            st.success(f"Upserted into DB: {vendor_id_in} / {supplier_id_in} → {tow_code_in}")
                            st.info("Click **♻️ Clear cache & re-run** to refresh mapping data.")
                    except Exception as e:
                        st.exception(e)

        st.subheader("Queued updates (data/updates.csv)")
        df_pending = load_pending_df()
        st.dataframe(df_pending, use_container_width=True, height=220)

        cA, cB, cC = st.columns([1, 1, 1])
        with cA:
            if st.button("Apply queued to SQLite (Upsert)"):
                try:
                    n = apply_pending_to_sqlite()
                    st.success(f"Applied {n} row(s) to DB.")
                    st.info("Click **♻️ Clear cache & re-run** above to see the changes.")
                except Exception as e:
                    st.exception(e)

        with cB:
            st.download_button(
                "Download queued CSV",
                data=df_pending.to_csv(index=False).encode("utf-8"),
                file_name="updates.csv",
                mime="text/csv",
                disabled=df_pending.empty,
            )

        with cC:
            if st.button("Clear queued CSV", type="secondary", disabled=df_pending.empty):
                try:
                    PENDING_CSV.unlink(missing_ok=True)
                    st.info("Cleared data/updates.csv.")
                except Exception as e:
                    st.exception(e)

        st.subheader("🧰 DB maintenance")
        uploaded_db = st.file_uploader("Replace data/crosswalk.db (upload a .db file)", type=["db"], key="db_upl")
        if uploaded_db is not None:
            _ensure_data_dir()
            DB_PATH.write_bytes(uploaded_db.getvalue())
            st.success(f"Replaced DB at {DB_PATH}.")
            st.info("Click **♻️ Clear cache & re-run** (top) to load the new DB.")

# =============================================================================
# Admin: Live search / inspect crosswalk
# =============================================================================
def _list_vendors():
    if not DB_PATH.exists():
        return []
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute("SELECT DISTINCT vendor_id FROM crosswalk ORDER BY vendor_id").fetchall()
        return [r[0] for r in rows if r[0] is not None and str(r[0]).strip()]
    except Exception:
        return []
    finally:
        con.close()

def _search_mappings(vendor_filter: str | None, supplier_q: str, exact: bool) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["vendor_id", "supplier_id", "tow_code"])
    con = sqlite3.connect(DB_PATH)
    try:
        cols = _detect_crosswalk_columns(con)
        supplier_col, tow_col, vendor_col = cols["supplier"], cols["tow"], cols["vendor"]
        select = f'SELECT {f""""{vendor_col}", """ if vendor_col else ""}"{supplier_col}" AS supplier_id, "{tow_col}" AS tow_code FROM crosswalk'
        clauses, params = [], []
        if vendor_col and vendor_filter and vendor_filter != "ALL":
            clauses.append(f'"{vendor_col}" = ?')
            params.append(vendor_filter)
        if supplier_q:
            if exact:
                clauses.append(f'"{supplier_col}" = ?')
                params.append(supplier_q)
            else:
                clauses.append(f'"{supplier_col}" LIKE ?')
                params.append(f"%{supplier_q}%")
        if clauses:
            select += " WHERE " + " AND ".join(clauses)
        select += f' ORDER BY {f""""{vendor_col}", """ if vendor_col else ""}"{supplier_col}" LIMIT 500'
        return pd.read_sql_query(select, con, params=params, dtype=str)
    finally:
        con.close()

with st.expander("🔎 Admin • Live search / inspect crosswalk", expanded=False):
    if not st.session_state.get("admin_pin_ok"):
        st.info("Unlock the Admin section first to use this panel.")
    else:
        vendors_list = _list_vendors()
        vendor_opt = ["ALL"] + vendors_list
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            pick_vendor = st.selectbox("Vendor", vendor_opt, index=0, key="search_vendor")
        with c2:
            supplier_q = st.text_input("supplier_id search", placeholder="exact or contains…")
        with c3:
            exact = st.checkbox("Exact", value=True)

        df_res = _search_mappings(pick_vendor, supplier_q.strip(), exact)
        st.caption(f"{len(df_res)} result(s) shown (max 500)")
        st.dataframe(df_res, use_container_width=True, height=260)

        if not df_res.empty:
            with st.form("prefill_form"):
                idx = st.number_input("Pick row # to prefill", min_value=0, max_value=len(df_res)-1, step=1, value=0)
                if st.form_submit_button("Prefill Admin form from row"):
                    row = df_res.iloc[int(idx)]
                    st.session_state["prefill_vendor_id"] = str(row.get("vendor_id", "") or "")
                    st.session_state["prefill_supplier_id"] = str(row.get("supplier_id", "") or "")
                    st.session_state["prefill_tow_code"] = str(row.get("tow_code", "") or "")
                    st.success("Prefilled. Scroll up to 'Add a single mapping'.")
