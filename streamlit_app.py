import os
import io
import sqlite3
from typing import List, Tuple

import pandas as pd
import streamlit as st

# ------------------------- BASIC SETTINGS -------------------------
st.set_page_config(page_title="Crosswalk Mapper", layout="wide")
ADMIN_PIN = os.getenv("CROSSWALK_ADMIN_PIN", "2468")
DB_PATH = os.path.join("data", "crosswalk.db")
CSV_FALLBACK = "crosswalk.csv"  # used if DB is missing (first run only)


# ------------------------- DB HELPERS -------------------------
def open_conn() -> sqlite3.Connection:
    """Open SQLite with sane defaults for Streamlit."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    Ensure crosswalk table + UNIQUE index exist.
    Table schema is intentionally simple: tow_code, supplier_id, vendor_id (TEXT).
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code     TEXT,
            supplier_id  TEXT,
            vendor_id    TEXT
        );
        """
    )
    # Unique index for ON CONFLICT upserts
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
        ON crosswalk(vendor_id, supplier_id);
        """
    )
    conn.commit()


def bootstrap_db_if_missing():
    """
    If DB doesn't exist but CSV is present, build a DB from CSV.
    Only runs on first-time setup.
    """
    if not os.path.exists(DB_PATH) and os.path.exists(CSV_FALLBACK):
        conn = open_conn()
        ensure_schema(conn)

        chunks = pd.read_csv(CSV_FALLBACK, dtype=str, chunksize=100_000)
        total = 0
        for chunk in chunks:
            chunk = normalize_crosswalk_df(chunk)
            chunk.to_sql("crosswalk", conn, if_exists="append", index=False)
            total += len(chunk)

        # Deduplicate using the unique index (will drop true duplicates)
        conn.execute(
            """
            DELETE FROM crosswalk
            WHERE rowid NOT IN (
              SELECT MIN(rowid) FROM crosswalk
              GROUP BY vendor_id, supplier_id
            );
            """
        )
        conn.commit()
        conn.close()


def normalize_crosswalk_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean incoming crosswalk data to three expected columns:
    ['tow_code', 'supplier_id', 'vendor_id'] all as strings, no spaces around.
    Accepts a variety of header spellings; raises if we can’t find what we need.
    """
    # Lowercase, strip, unify column names
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Candidate columns
    tow_cols = ["tow", "tow_code", "towcode", "towkod", "tow_kod", "towkod"]
    sup_cols = ["supplier_id", "supplier", "supplier code", "supplier_code", "sup", "sup_code", "customs code", "customs_code"]
    ven_cols = ["vendor_id", "vendor", "vendor code", "vendor_code"]

    def pick(cols: List[str]) -> str:
        for c in cols:
            if c in df.columns:
                return c
        return ""

    tow = pick(tow_cols)
    sup = pick(sup_cols)
    ven = pick(ven_cols)  # vendor optional in raw csv, but we prefer to have it

    if not sup:
        raise ValueError("Could not find SUPPLIER code column in crosswalk CSV.")
    if not tow:
        raise ValueError("Could not find TOW code column in crosswalk CSV.")
    if not ven:
        ven = "vendor_id"
        df[ven] = ""  # placeholder if CSV had no vendor info

    out = pd.DataFrame(
        {
            "tow_code": df[tow].astype(str).str.strip(),
            "supplier_id": df[sup].astype(str).str.strip(),
            "vendor_id": df[ven].astype(str).str.strip(),
        }
    )
    return out


def get_vendors(conn: sqlite3.Connection) -> List[str]:
    vendors = pd.read_sql("SELECT DISTINCT vendor_id FROM crosswalk ORDER BY vendor_id", conn)
    vals = vendors["vendor_id"].fillna("").tolist()
    return [v for v in vals if v]  # drop blanks


def upsert_mapping(conn: sqlite3.Connection, vendor_id: str, supplier_id: str, tow_code: str) -> Tuple[bool, str]:
    """
    Upsert (vendor_id, supplier_id) -> tow_code.
    Returns (ok, message).
    """
    try:
        conn.execute(
            """
            INSERT INTO crosswalk(tow_code, supplier_id, vendor_id)
            VALUES(?, ?, ?)
            ON CONFLICT(vendor_id, supplier_id)
            DO UPDATE SET tow_code = excluded.tow_code;
            """,
            (tow_code.strip(), supplier_id.strip(), vendor_id.strip()),
        )
        conn.commit()
        return True, "Upsert OK."
    except Exception as e:
        return False, f"Upsert failed: {e}"


def live_search(
    conn: sqlite3.Connection, vendor_id: str, supplier_like: str, tow_like: str, limit: int = 500
) -> pd.DataFrame:
    """
    Live search over crosswalk with contains filters.
    """
    vendor_q = "" if vendor_id == "ALL" else "AND vendor_id = ?"
    params: List[str] = []
    sql = f"""
    SELECT tow_code, supplier_id, vendor_id
    FROM crosswalk
    WHERE 1=1
      {vendor_q}
    """
    if vendor_id != "ALL":
        params.append(vendor_id)

    if supplier_like.strip():
        sql += " AND supplier_id LIKE ?"
        params.append(f"%{supplier_like.strip()}%")

    if tow_like.strip():
        sql += " AND tow_code LIKE ?"
        params.append(f"%{tow_like.strip()}%")

    sql += " ORDER BY vendor_id, supplier_id LIMIT ?"
    params.append(limit)

    return pd.read_sql(sql, conn, params=params)


# ------------------------- UI HELPERS -------------------------
def banner_ok(db_path: str, rows: int, vendors: int):
    st.success(
        f"Database ready at **{db_path}** • rows: **{rows:,}** • vendors: **{vendors}**"
    )


def read_invoice(upload) -> pd.DataFrame:
    """
    Read user’s invoice (CSV/XLSX) into DataFrame[str].
    """
    if upload.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(upload, dtype=str)
    else:
        # Try CSV with a couple of separators
        upload.seek(0)
        raw = upload.read()
        # guess sep (common: ';' or ',')
        try:
            df = pd.read_csv(io.BytesIO(raw), dtype=str, sep=";")
            if df.shape[1] == 1:  # probably wrong sep
                df = pd.read_csv(io.BytesIO(raw), dtype=str)
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), dtype=str)
    df = df.fillna("")
    return df


def guess_supplier_col(cols: List[str]) -> str:
    low = [c.lower() for c in cols]
    candidates = [
        "supplier_id",
        "supplier",
        "supplier code",
        "supplier_code",
        "sup_code",
        "customs code",
        "customs_code",
        "code",
    ]
    for c in candidates:
        if c in low:
            return cols[low.index(c)]
    # fallback: first text-like column
    return cols[0] if cols else ""


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            df.to_excel(xw, index=False, sheet_name=sheet_name)
        return buf.getvalue()


# ------------------------- APP -------------------------
def main():
    st.title("Supplier → TOW Mapper (SQLite)")

    # Ensure DB exists
    bootstrap_db_if_missing()

    # Connect + ensure schema/index
    conn = open_conn()
    ensure_schema(conn)

    # Quick DB stats
    rows = pd.read_sql("SELECT COUNT(*) n FROM crosswalk", conn)["n"].iat[0]
    vlist = get_vendors(conn)
    banner_ok(DB_PATH, rows, len(vlist))

    st.write("---")

    # 1) Upload invoice
    st.header("2) Upload supplier invoice (Excel / CSV)")
    up = st.file_uploader("Drag & drop or Browse", type=["csv", "xlsx", "xls"])
    invoice_df = None

    if up:
        try:
            invoice_df = read_invoice(up)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return

        st.caption(
            f"Preview • Rows: **{len(invoice_df):,}** • Columns: {', '.join(invoice_df.columns[:8])}"
        )
        st.dataframe(invoice_df.head(10), use_container_width=True)

    # 2) Pick vendor + supplier-code column
    st.header("3) Map to TOW")

    colA, colB = st.columns([1, 2])
    with colA:
        vendor_opt = ["ALL"] + vlist
        pick_vendor = st.selectbox("Vendor", vendor_opt, index=0)

    supplier_col = ""
    if invoice_df is not None and not invoice_df.empty:
        supplier_col = st.selectbox(
            "Which column contains the SUPPLIER code?",
            options=list(invoice_df.columns),
            index=max(0, list(invoice_df.columns).index(guess_supplier_col(list(invoice_df.columns)))),
        )

    # 3) Run mapping
    if invoice_df is not None and supplier_col:
        st.button("Run mapping", key="run_mapping")
        # Normalize invoice supplier column (string)
        inv = invoice_df.copy()
        inv[supplier_col] = inv[supplier_col].astype(str).str.strip()

        # Build crosswalk view (depending on vendor selection)
        if pick_vendor == "ALL":
            cw = pd.read_sql("SELECT tow_code, supplier_id, vendor_id FROM crosswalk", conn)
        else:
            cw = pd.read_sql(
                "SELECT tow_code, supplier_id, vendor_id FROM crosswalk WHERE vendor_id = ?",
                conn,
                params=[pick_vendor],
            )

        # Merge: invoice rows → match by supplier_id and (optional) vendor filter used above
        merged = inv.merge(
            cw.rename(columns={"supplier_id": supplier_col}),
            how="left",
            on=supplier_col,
            suffixes=("", "_cw"),
        )

        matched = merged[merged["tow_code"].notna()].copy()
        unmatched = merged[merged["tow_code"].isna()].copy()

        st.success(f"Mapping complete → matched: **{len(matched):,}** | unmatched: **{len(unmatched):,}**")

        with st.expander("Preview: Matched (first 200 rows)", expanded=False):
            st.dataframe(matched.head(200), use_container_width=True)
            st.download_button(
                "Download Matched (Excel)",
                to_excel_bytes(matched),
                file_name="matched.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with st.expander("Preview: Unmatched (first 200 rows)", expanded=False):
            st.dataframe(unmatched.head(200), use_container_width=True)
            st.download_button(
                "Download Unmatched (Excel)",
                to_excel_bytes(unmatched),
                file_name="unmatched.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ------------------------- ADMIN -------------------------
    st.write("---")
    with st.expander("🔐 Admin • Live search / Add mapping (direct upsert)", expanded=False):
        pin_col, _ = st.columns([1, 4])
        with pin_col:
            pin = st.text_input("Admin PIN", type="password")
            unlocked = st.button("Unlock")

        if unlocked:
            st.session_state["admin_unlocked"] = (pin == ADMIN_PIN)

        if st.session_state.get("admin_unlocked"):
            st.success("Admin unlocked.")

            st.subheader("Add a single mapping (direct UPSERT to SQLite)")
            c1, c2, c3 = st.columns(3)
            with c1:
                vendor_id = st.text_input("vendor_id", value="")
            with c2:
                supplier_id = st.text_input("supplier_id", value="")
            with c3:
                tow_code = st.text_input("tow_code", value="")

            if st.button("Add (Direct to SQLite)"):
                ok, msg = upsert_mapping(conn, vendor_id, supplier_id, tow_code)
                (st.success if ok else st.error)(msg)

            st.write("---")
            st.subheader("Live search / inspect crosswalk")

            vendor_pick = st.selectbox("Vendor", ["ALL"] + vlist)
            s_like = st.text_input("supplier_id contains …", value="")
            t_like = st.text_input("tow_code contains …", value="")
            limit = st.number_input("Rows to show (max 2000)", min_value=50, max_value=2000, value=500, step=50)

            if st.button("Search"):
                df = live_search(conn, vendor_pick, s_like, t_like, int(limit))
                st.caption(f"{len(df)} result(s) shown")
                st.dataframe(df, use_container_width=True)

        else:
            st.info("Enter PIN and press Unlock to access admin features.")

    # Footer
    st.caption(
        f"DB: **{DB_PATH}**"
    )


if __name__ == "__main__":
    main()
