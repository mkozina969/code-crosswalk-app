from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import streamlit as st

# --- DB helpers (ADD THIS BLOCK NEAR THE TOP) -------------------------------
from pathlib import Path
import sqlite3

DB_PATH = Path("data/crosswalk.db")

def open_db() -> sqlite3.Connection:
    """
    Always open the single source of truth DB and guarantee schema + index.
    Safe to call many times; CREATE IF NOT EXISTS is idempotent.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)

    # Ensure table
    con.execute("""
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code    TEXT,
            supplier_id TEXT,
            vendor_id   TEXT
        )
    """)

    # Ensure UNIQUE index used by ON CONFLICT clause (very important)
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
        ON crosswalk(vendor_id, supplier_id)
    """)

    con.commit()
    return con


def upsert_mapping(con: sqlite3.Connection, vendor_id: str, supplier_id: str, tow_code: str) -> None:
    """
    Insert or update mapping for (vendor_id, supplier_id) with tow_code.
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
# ---------------------------------------------------------------------------

# -----------------------------
# Paths / constants
# -----------------------------
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "crosswalk.db"
CSV_FALLBACK = Path("crosswalk.csv")  # used to build DB if DB is missing

# -----------------------------
# One-time schema bootstrap
# -----------------------------
def ensure_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code     TEXT NOT NULL,
            supplier_id  TEXT NOT NULL,
            vendor_id    TEXT NOT NULL,
            PRIMARY KEY(tow_code, supplier_id, vendor_id)
        )
        """
    )
    con.commit()
    con.close()


# -----------------------------
# CSV -> DB builder (fallback)
# -----------------------------
def _pick_col(cols: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols_l = [c.lower().strip() for c in cols]
    for want in candidates:
        try:
            idx = cols_l.index(want.lower().strip())
            return list(cols)[idx]
        except ValueError:
            continue
    return None


def build_db_from_csv(csv_path: Path, db_path: Path, chunksize: int = 200_000) -> Tuple[int, int]:
    """
    Build the SQLite DB from a CSV file.
    Returns (rows_inserted, vendor_count).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    ensure_schema(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Try Python engine for flexible delimiters
    # We normalize headers to tow_code, supplier_id, vendor_id if possible
    total = 0
    vendor_set = set()

    # Detect delimiter quickly
    try:
        probe = pd.read_csv(csv_path, nrows=500, engine="python")
        delim = None  # pandas auto-detected
    except Exception:
        # fallback to semicolon
        probe = pd.read_csv(csv_path, nrows=500, engine="python", sep=";")
        delim = ";"

    # Normalize columns
    tow_col = _pick_col(probe.columns, ["tow", "tow_code", "towkod", "tow_kod"])
    sup_col = _pick_col(probe.columns, ["supplier_id", "supplier", "vendor code", "supplier_code"])
    ven_col = _pick_col(probe.columns, ["vendor_id", "vendor", "vendor navi", "vendor_navi", "navi_vendor"])

    if not (tow_col and sup_col and ven_col):
        raise ValueError(
            "crosswalk.csv must contain columns for TOW, supplier_id and vendor_id. "
            f"Detected: tow={tow_col}, supplier_id={sup_col}, vendor_id={ven_col}"
        )

    # Stream chunks
    read_kw = dict(engine="python")
    if delim:
        read_kw["sep"] = delim

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, **read_kw):
        # keep only required and normalize
        c = chunk[[tow_col, sup_col, ven_col]].copy()
        c.columns = ["tow_code", "supplier_id", "vendor_id"]
        c["tow_code"] = c["tow_code"].astype(str).str.strip().str.upper()
        c["supplier_id"] = c["supplier_id"].astype(str).str.strip()
        c["vendor_id"] = c["vendor_id"].astype(str).str.strip().str.upper()

        # Track vendors
        vendor_set.update(c["vendor_id"].unique().tolist())

        # UPSERT
        rows = list(c.itertuples(index=False, name=None))
        cur.executemany(
            """
            INSERT INTO crosswalk (tow_code, supplier_id, vendor_id)
            VALUES (?, ?, ?)
            ON CONFLICT(tow_code, supplier_id, vendor_id) DO NOTHING
            """,
            rows,
        )
        con.commit()
        total += len(rows)

    con.close()
    return total, len(vendor_set)


# -----------------------------
# Cached loaders (with cache key)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_crosswalk_df(_rev: int) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT tow_code, supplier_id, vendor_id FROM crosswalk",
        con,
        dtype={"tow_code": "string", "supplier_id": "string", "vendor_id": "string"},
    )
    con.close()
    # Normalize
    df["tow_code"] = df["tow_code"].str.upper()
    df["vendor_id"] = df["vendor_id"].str.upper()
    return df


def upsert_mapping(tow_code: str, supplier_id: str, vendor_id: str) -> int:
    """UPSERT one row; returns 1 if inserted, 0 if already present."""
    ensure_schema(DB_PATH)
    tow = str(tow_code).strip().upper()
    sup = str(supplier_id).strip()
    ven = str(vendor_id).strip().upper()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO crosswalk (tow_code, supplier_id, vendor_id)
        VALUES (?, ?, ?)
        ON CONFLICT(tow_code, supplier_id, vendor_id) DO NOTHING
        """,
        (tow, sup, ven),
    )
    n = cur.rowcount
    con.commit()
    con.close()
    return n


def refresh_cache():
    st.session_state["db_rev"] = st.session_state.get("db_rev", 0) + 1


# -----------------------------
# Invoice helpers
# -----------------------------
def read_invoice(upload) -> pd.DataFrame:
    name = (upload.name or "").lower()
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(upload, engine="openpyxl")
    else:
        # CSV; try flexible parser first, then fallback to semicolon
        try:
            df = pd.read_csv(upload, engine="python")
        except Exception:
            upload.seek(0)
            df = pd.read_csv(upload, engine="python", sep=";")
    # Trim column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def guess_supplier_col(cols: Iterable[str]) -> str | None:
    return _pick_col(
        cols,
        [
            "supplier_id",
            "supplier",
            "supplier_code",
            "vendor code",
            "customs code",
            "code",
            "ean",
        ],
    )


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Supplier → TOW Mapper", layout="wide")

st.title("Supplier → TOW Mapper (SQLite)")
with st.expander("How to use", expanded=True):
    st.markdown(
        """
1. The app opens the crosswalk **SQLite DB** from `data/crosswalk.db` (or builds it from `crosswalk.csv` if missing).
2. Upload your **supplier invoice** (Excel/CSV).
3. Pick the **Vendor** and the invoice's **Supplier Code** column.
4. Download **Matched / Unmatched**.
5. Use **Admin** to add missing rows directly to SQLite; changes appear **instantly** in Live search.
        """
    )

# cache key
if "db_rev" not in st.session_state:
    st.session_state["db_rev"] = 0

# Ensure DB; build from CSV if needed
ensure_schema(DB_PATH)
built_note = ""
if not DB_PATH.exists():
    if CSV_FALLBACK.exists():
        with st.spinner("Building SQLite DB from crosswalk.csv …"):
            total, vcount = build_db_from_csv(CSV_FALLBACK, DB_PATH)
        built_note = f" — built from CSV ({total:,} rows, {vcount} vendors)"
    else:
        built_note = " — (empty DB created; add mappings in Admin)"

# Load CW
cw = load_crosswalk_df(st.session_state["db_rev"])
vendor_count = cw["vendor_id"].nunique()
st.success(f"Crosswalk loaded: {len(cw):,} rows, {vendor_count} vendors{built_note}")

# Vendor picker (top)
vendor_opt = ["ALL"] + sorted(cw["vendor_id"].dropna().unique().tolist())
pick_vendor = st.selectbox("Vendor", vendor_opt, index=0, key="vendor_select_top")

st.markdown("---")
st.header("2) Upload supplier invoice (Excel / CSV)")
upload = st.file_uploader("Drag and drop or Browse", type=["xlsx", "xls", "csv"])
inv_df = None
sup_col = None

if upload is not None:
    try:
        inv_df = read_invoice(upload)
        st.caption(
            f"Rows: {len(inv_df):,} | Columns: {', '.join([f'`{c}`' for c in inv_df.columns])}"
        )
        st.dataframe(inv_df.head(10), use_container_width=True, height=250)

        # Which column contains supplier code?
        guess = guess_supplier_col(inv_df.columns)
        sup_col = st.selectbox(
            "Which column contains the SUPPLIER code?",
            inv_df.columns.tolist(),
            index=(inv_df.columns.tolist().index(guess) if guess in inv_df.columns else 0),
            key="supplier_col_pick",
        )
    except Exception as e:
        st.error(f"Failed to load invoice: {e}")

st.markdown("---")
st.header("3) Map to TOW")

if st.button("Run mapping", disabled=inv_df is None or sup_col is None, type="primary"):
    if inv_df is None or sup_col is None:
        st.warning("Please upload an invoice and choose the supplier code column.")
    else:
        # Prepare input
        work = inv_df.copy()
        work["_supplier_id_norm"] = work[sup_col].astype(str).str.strip()

        # Filter crosswalk by vendor
        cwv = cw if pick_vendor == "ALL" else cw[cw["vendor_id"] == pick_vendor]

        # Join
        merged = work.merge(
            cwv[["supplier_id", "tow_code", "vendor_id"]],
            left_on="_supplier_id_norm",
            right_on="supplier_id",
            how="left",
        )
        matched = merged[merged["tow_code"].notna()].copy()
        unmatched = merged[merged["tow_code"].isna()].copy()

        st.success(f"Mapping complete → matched: {len(matched):,} | unmatched: {len(unmatched):,}")

        with st.expander("Preview: Matched (first 200 rows)", expanded=False):
            st.dataframe(matched.head(200), use_container_width=True, height=300)
        with st.expander("Preview: Unmatched (first 200 rows)", expanded=False):
            st.dataframe(unmatched.head(200), use_container_width=True, height=300)

        # Download both sheets as one xlsx
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            matched.to_excel(xw, index=False, sheet_name="Matched")
            unmatched.to_excel(xw, index=False, sheet_name="Unmatched")
        st.download_button(
            "Download Excel (Matched + Unmatched)",
            data=buf.getvalue(),
            file_name="mapping_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.markdown("---")
with st.expander("🔐 Admin • Add / Live search / Apply mappings", expanded=False):
    # PIN unlock (very lightweight)
    pin_ok = st.session_state.get("pin_ok", False)
    colp1, colp2 = st.columns([1, 4])
    with colp1:
        pin_mask = st.text_input("Admin PIN", type="password", key="admin_pin")
        if st.button("Unlock"):
            st.session_state["pin_ok"] = (pin_mask or "").strip() != ""  # simple gate
            pin_ok = st.session_state["pin_ok"]
    if not pin_ok:
        st.info("Enter any PIN to unlock admin tools (demo gate).")
        st.stop()

    st.success("Admin unlocked.")

    st.caption(
        f"DB: `{DB_PATH.as_posix()}`  •  Current rows: {len(cw):,}  •  Vendors: {vendor_count}"
    )

    st.subheader("Add a single mapping (direct UPSERT to SQLite)")

    a1, a2, a3 = st.columns(3)
    with a1:
        admin_vendor = st.text_input("vendor_id", placeholder="e.g. DOB0000025", key="adm_vendor")
    with a2:
        admin_supplier = st.text_input("supplier_id", placeholder="e.g. 0986356023", key="adm_supplier")
    with a3:
        admin_tow = st.text_input("tow_code", placeholder="e.g. 200183", key="adm_tow")

    if st.button("Add (Direct to SQLite)"):
        if not (admin_vendor and admin_supplier and admin_tow):
            st.warning("Please fill vendor_id, supplier_id and tow_code.")
        else:
            try:
                n = upsert_mapping(admin_tow, admin_supplier, admin_vendor)
                refresh_cache()  # bump cache key
                if n == 1:
                    st.success("Inserted.")
                else:
                    st.info("Row already exists (no change).")
                st.rerun()  # immediate refresh so Live search sees it
            except Exception as e:
                st.error(f"Insert failed: {e}")

    st.markdown("---")
    st.subheader("Live search / inspect crosswalk")

    df_live = load_crosswalk_df(st.session_state["db_rev"])
    lv_vendor = st.selectbox(
        "Vendor", ["ALL"] + sorted(df_live["vendor_id"].dropna().unique().tolist()),
        index=0, key="vendor_select_live"
    )
    q_supplier = st.text_input("supplier_id contains …", key="live_q")

    view = df_live
    if lv_vendor != "ALL":
        view = view[view["vendor_id"] == lv_vendor]
    if q_supplier:
        view = view[view["supplier_id"].str.contains(q_supplier, case=False, na=False)]

    st.dataframe(view.head(500), use_container_width=True, height=400)
