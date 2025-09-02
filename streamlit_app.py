from __future__ import annotations

import os
import io
import csv
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st


# --------------------------
# Paths & basic config
# --------------------------
APP_ROOT = Path(__file__).resolve().parent

# Allow override via env var; otherwise try common locations
CANDIDATE_DB_PATHS = [
    os.environ.get("CROSSWALK_DB") or "",
    APP_ROOT / "data" / "crosswalk.db",
    APP_ROOT.parent / "data" / "crosswalk.db",
    Path("data/crosswalk.db"),
]

UPDATES_CSV = APP_ROOT / "data" / "updates.csv"

st.set_page_config(page_title="Supplier → TOW Mapper (SQLite)", layout="wide")


# --------------------------
# DB helpers
# --------------------------
def _choose_db_path() -> Path:
    for p in CANDIDATE_DB_PATHS:
        if not p:
            continue
        p = Path(p)
        if p.exists():
            return p
    # last resort: show where we looked & fail visibly
    tried = [str(Path(x)) for x in CANDIDATE_DB_PATHS if x]
    st.error(
        "Could not locate crosswalk database.\n\n"
        "Tried: \n- " + "\n- ".join(tried)
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def get_db_path() -> Path:
    return _choose_db_path()


def connect_db() -> sqlite3.Connection:
    db_path = get_db_path()
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    # make sure the unique index exists so upserts never fail
    ensure_unique_index(con)
    return con


def ensure_unique_index(con: sqlite3.Connection) -> None:
    # Unique index on (vendor_id, supplier_id) ensures ON CONFLICT works
    con.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crosswalk_vs
        ON crosswalk(vendor_id, supplier_id);
        """
    )
    con.commit()


@st.cache_data(show_spinner=False)
def get_db_counts() -> Tuple[int, int]:
    con = connect_db()
    try:
        total = con.execute("SELECT COUNT(*) FROM crosswalk").fetchone()[0]
        vendors = con.execute("SELECT COUNT(DISTINCT vendor_id) FROM crosswalk").fetchone()[0]
        return total, vendors
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def get_vendor_list() -> List[str]:
    con = connect_db()
    try:
        rows = con.execute(
            "SELECT DISTINCT vendor_id FROM crosswalk ORDER BY vendor_id"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def upsert_mapping(vendor_id: str, supplier_id: str, tow_code: str) -> int:
    con = connect_db()
    try:
        cur = con.execute(
            """
            INSERT INTO crosswalk(tow_code, supplier_id, vendor_id)
            VALUES(?, ?, ?)
            ON CONFLICT(vendor_id, supplier_id)
            DO UPDATE SET tow_code = excluded.tow_code;
            """,
            (tow_code, supplier_id, vendor_id),
        )
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def live_search_crosswalk(
    vendor: str | None, supplier_contains: str | None, limit: int = 500
) -> pd.DataFrame:
    con = connect_db()
    try:
        if vendor and vendor != "ALL" and supplier_contains:
            q = """
                SELECT tow_code, supplier_id, vendor_id
                FROM crosswalk
                WHERE vendor_id = ?
                  AND supplier_id LIKE ?
                ORDER BY supplier_id
                LIMIT ?
            """
            params = (vendor, f"%{supplier_contains}%", limit)
        elif vendor and vendor != "ALL":
            q = """
                SELECT tow_code, supplier_id, vendor_id
                FROM crosswalk
                WHERE vendor_id = ?
                ORDER BY supplier_id
                LIMIT ?
            """
            params = (vendor, limit)
        elif supplier_contains:
            q = """
                SELECT tow_code, supplier_id, vendor_id
                FROM crosswalk
                WHERE supplier_id LIKE ?
                ORDER BY vendor_id, supplier_id
                LIMIT ?
            """
            params = (f"%{supplier_contains}%", limit)
        else:
            q = """
                SELECT tow_code, supplier_id, vendor_id
                FROM crosswalk
                ORDER BY vendor_id, supplier_id
                LIMIT ?
            """
            params = (limit,)

        df = pd.read_sql_query(q, con, params=params)
        return df
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def get_vendor_mapping(vendor_id: str) -> pd.DataFrame:
    """Supplier → TOW mapping for a single vendor."""
    con = connect_db()
    try:
        df = pd.read_sql_query(
            "SELECT supplier_id, tow_code FROM crosswalk WHERE vendor_id = ?",
            con,
            params=(vendor_id,),
        )
        return df
    finally:
        con.close()


# --------------------------
# CSV/Excel helpers
# --------------------------
def _read_csv_flex(file) -> pd.DataFrame:
    """Robust CSV reader – tries sep=',' then ';' (common in EU exports)."""
    file.seek(0)
    try:
        df = pd.read_csv(file, dtype=str, engine="python")
        if df.shape[1] == 1:
            file.seek(0)
            df = pd.read_csv(file, dtype=str, sep=";", engine="python")
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, dtype=str, sep=";", engine="python")

    return df


def read_any_table(upload) -> pd.DataFrame:
    name = (upload.name or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(upload, dtype=str)
    else:
        # assume CSV
        df = _read_csv_flex(upload)
    return df.convert_dtypes()


# --------------------------
# Queue helpers (CSV)
# --------------------------
def queue_append(vendor_id: str, supplier_id: str, tow_code: str) -> None:
    UPDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not UPDATES_CSV.exists()
    with UPDATES_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["vendor_id", "supplier_id", "tow_code"])
        w.writerow([vendor_id, supplier_id, tow_code])


def queue_read() -> pd.DataFrame:
    if not UPDATES_CSV.exists():
        return pd.DataFrame(columns=["vendor_id", "supplier_id", "tow_code"])
    return pd.read_csv(UPDATES_CSV, dtype=str)


def queue_clear() -> None:
    if UPDATES_CSV.exists():
        UPDATES_CSV.unlink()


def queue_apply() -> Tuple[int, int]:
    """Apply queued updates to SQLite (upsert). Returns (applied, errors)."""
    df = queue_read()
    if df.empty:
        return 0, 0
    ok = 0
    bad = 0
    for _, row in df.iterrows():
        try:
            upsert_mapping(
                vendor_id=str(row["vendor_id"]),
                supplier_id=str(row["supplier_id"]),
                tow_code=str(row["tow_code"]),
            )
            ok += 1
        except Exception:
            bad += 1
    return ok, bad


# --------------------------
# UI helpers
# --------------------------
def banner():
    db_path = get_db_path()
    total, vendors = get_db_counts()
    st.success(
        f"Database ready at **{db_path}** • rows: **{total:,}** • vendors: **{vendors}**"
    )


def choose_supplier_col(df: pd.DataFrame) -> str:
    # heuristic: pick probable columns, else let user choose any
    candidates = [
        "supplier_id",
        "supplierid",
        "supplier_code",
        "suppliercode",
        "supplier_ic",
        "supplier",
        "code",
        "id",
    ]
    cols = list(df.columns)
    default = 0
    for i, c in enumerate(cols):
        if c.strip().lower() in candidates:
            default = i
            break
    col = st.selectbox("Which column contains the SUPPLIER code?", cols, index=default, key="supplier_col_pick")
    return col


def table_download_button(df1: pd.DataFrame, df2: pd.DataFrame, label: str):
    """Download matched + unmatched to a single Excel file."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        df1.to_excel(xw, index=False, sheet_name="Matched")
        df2.to_excel(xw, index=False, sheet_name="Unmatched")
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name="mapping_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_mapping_result",
    )


# --------------------------
# App
# --------------------------
def main():
    st.title("Supplier → TOW Mapper (SQLite)")
    banner()
    st.divider()

    # ----------------------
    # 2) Upload supplier invoice (Excel / CSV)
    # ----------------------
    st.header("2) Upload supplier invoice (Excel / CSV)")

    upload = st.file_uploader(
        "Drag & drop or Browse", type=["csv", "xlsx", "xls"], key="uploader"
    )

    if upload is not None:
        try:
            df = read_any_table(upload)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        with st.expander("Preview:", expanded=True):
            st.write(df.head(10))
            st.caption(f"Rows: {len(df):,} | Columns: {', '.join(df.columns)}")

        supplier_col = choose_supplier_col(df)
        st.session_state["upload_df"] = df
        st.session_state["supplier_col"] = supplier_col
    else:
        st.session_state.pop("upload_df", None)
        st.session_state.pop("supplier_col", None)

    st.divider()

    # ----------------------
    # 3) Map to TOW
    # ----------------------
    st.header("3) Map to TOW")

    vendors = ["ALL"] + get_vendor_list()
    pick_vendor = st.selectbox("Vendor", vendors, index=0, key="map_vendor")

    # Only run mapping if we have upload + a specific vendor chosen
    if "upload_df" in st.session_state and pick_vendor != "ALL":
        df_up = st.session_state["upload_df"].copy()
        supplier_col = st.session_state["supplier_col"]

        # Create a clean supplier_id series
        df_up["supplier_id"] = df_up[supplier_col].astype(str).str.strip()

        if st.button("Run mapping", type="primary", key="run_mapping"):
            vendor_map = get_vendor_mapping(pick_vendor)
            merged = df_up.merge(vendor_map, how="left", on="supplier_id")
            matched = merged[merged["tow_code"].notna()].copy()
            unmatched = merged[merged["tow_code"].isna()].copy()

            st.success(
                f"Mapping complete → **matched:** {len(matched):,} • **unmatched:** {len(unmatched):,}"
            )

            with st.expander("Preview: Matched (first 200 rows)", expanded=False):
                st.dataframe(matched.head(200), use_container_width=True)

            with st.expander("Preview: Unmatched (first 200 rows)", expanded=False):
                st.dataframe(unmatched.head(200), use_container_width=True)

            table_download_button(matched, unmatched, "Download Excel (Matched + Unmatched)")

    # ----------------------
    # Admin tools
    # ----------------------
    st.divider()
    with st.expander("🔐 Admin • Live search / Add mapping (direct upsert)"):
        col_pin, _ = st.columns([1, 3])
        with col_pin:
            pin = st.text_input("Admin PIN", type="password", key="admin_pin")
            ok = st.button("Unlock", key="unlock_admin")

        if ok and pin == "2468":
            st.session_state["admin_unlocked"] = True
        elif ok:
            st.warning("Wrong PIN")

        if st.session_state.get("admin_unlocked"):
            st.success("Admin unlocked.")

            # --- Add single mapping
            st.subheader("Add a single mapping (direct UPSERT to SQLite)")
            with st.form("add_single_form", clear_on_submit=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    v = st.text_input("vendor_id", placeholder="e.g. DOB0000025", key="adm_vendor")
                with c2:
                    s = st.text_input("supplier_id", placeholder="e.g. 0983656023", key="adm_supplier")
                with c3:
                    t = st.text_input("tow_code", placeholder="e.g. 200183", key="adm_tow")

                on_submit = st.form_submit_button("Add (Direct to SQLite)")
                queue_submit = st.form_submit_button("Add to queue (data/updates.csv)")

            if on_submit and v and s and t:
                try:
                    n = upsert_mapping(v, s, t)
                    st.success(f"Upsert complete (rows affected: {n}).")
                    # refresh counts
                    get_db_counts.clear()
                except Exception as e:
                    st.error(f"Insert failed: {e}")

            if queue_submit and v and s and t:
                queue_append(v, s, t)
                st.success(f"Queued: {v}, {s} → {t}")

            # --- Queue view & actions
            st.subheader("Queued updates (data/updates.csv)")
            qdf = queue_read()
            st.dataframe(qdf, use_container_width=True, height=260)

            cqa, cqb, cqc = st.columns([1, 1, 2])
            with cqa:
                if st.button("Apply queued to SQLite (Upsert)"):
                    okn, badn = queue_apply()
                    st.success(f"Applied: {okn} • Errors: {badn}")
                    # Refresh counts
                    get_db_counts.clear()
            with cqb:
                if st.button("Clear queued CSV"):
                    queue_clear()
                    st.info("Queue cleared.")
            with cqc:
                if not qdf.empty:
                    out = io.BytesIO()
                    qdf.to_csv(out, index=False)
                    st.download_button(
                        "Download queued CSV",
                        data=out.getvalue(),
                        file_name="updates.csv",
                        mime="text/csv",
                        key="dl_queue",
                    )

            # --- Live Search
            st.subheader("Live search / inspect crosswalk")
            c1, c2 = st.columns([1, 2])
            with c1:
                vpick = st.selectbox("Vendor", ["ALL"] + get_vendor_list(), key="live_vendor")
            with c2:
                ssub = st.text_input("supplier_id contains ...", key="live_sup_sub")

            df_live = live_search_crosswalk(vpick, ssub, limit=500)
            st.caption("0 results shown (max 500)") if df_live.empty else st.caption(
                f"{len(df_live):,} results (max 500)"
            )
            st.dataframe(df_live, use_container_width=True, height=320)

        else:
            st.info("Enter PIN to unlock admin tools. (Default demo PIN: 2468)")


if __name__ == "__main__":
    main()
