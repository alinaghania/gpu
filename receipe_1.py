"""
RECIPE 1 — EVENTS SAMPLED (FAST)
----------------------------------------------
But: construire un tout petit échantillon récent pour smoke-test end-to-end,
en maximisant la vitesse (pas de rescans par mois, stop early dès quota atteint).

Entrée : BASE_SCORE_COMPLETE_prepared
Sortie : events_sampled

Colonnes de sortie :
- CLIENT_ID
- DATE_EVENT (timestamp naïf)
- PRODUCT_CODE
- PARTITION_MONTH (YYYY-MM)
"""

import sys, logging
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import dataiku

# ===================== PARAMS =========================
DATASET_MAIN        = "BASE_SCORE_COMPLETE_prepared"
OUTPUT_DATASET_NAME = "events_sampled"

# Colonnes source
CLIENT_ID_COL    = "NUMTECPRS"
TIME_COL         = "DATMAJ"
PRODUCT_COL      = "SOUSCRIPTION_PRODUIT_1M"
EXTRA_EVENT_COLS = []  # ex: ["CANAL", "FAMILLE"]

# Échantillon minuscule et récent
LIMIT_ROWS         = 400          # tiny
MONTHS_BACK_TARGET = 2            # 1-2 mois récents suffisent pour un test
MIN_CLIENT_MONTHS  = 1            # tolérant
EXCLUDE_ON_EVENTS  = False        # <<< pas d'exclusion pour aller VITE

# Lecture streaming
CHUNKSIZE          = 200_000      # gros chunks pour réduire overhead
RANDOM_SEED        = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("recipe1_fast")


def _to_month_label(dt_series: pd.Series) -> pd.Series:
    return pd.to_datetime(dt_series, errors="coerce", utc=True).dt.tz_convert(None).dt.to_period("M").astype(str)


def _list_partitions_if_any(ds: dataiku.Dataset) -> List[str]:
    try:
        parts = ds.list_partitions()
        parts = sorted(parts, reverse=True)
        return parts
    except Exception:
        return []


def _find_max_month_by_scan(ds: dataiku.Dataset) -> Optional[str]:
    """1 pass légère (TIME_COL only) pour trouver le mois max si pas de partitions."""
    max_month = None
    for ch in ds.iter_dataframes(columns=[TIME_COL], chunksize=CHUNKSIZE,
                                 parse_dates=False, infer_with_pandas=True):
        m = _to_month_label(ch[TIME_COL]).dropna()
        if not m.empty:
            mm = m.max()
            if (max_month is None) or (mm > max_month):
                max_month = mm
    return max_month


def _month_minus(month_str: str, k: int) -> str:
    # month_str "YYYY-MM" -> subtract k months
    p = pd.Period(month_str, freq="M")
    return (p - k).strftime("%Y-%m")


def _stream_filter_last_months(ds: dataiku.Dataset, min_month_inclusive: str,
                               limit_rows: int, exclude_class: bool) -> pd.DataFrame:
    """2e pass : filtre en streaming sur les derniers mois, stop early dès LIMIT_ROWS atteint."""
    cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)
    out = []
    total = 0

    for ch in ds.iter_dataframes(columns=cols, chunksize=CHUNKSIZE,
                                 parse_dates=False, infer_with_pandas=True):
        # parse date & month
        ch["DATE_EVENT"] = pd.to_datetime(ch[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
        ch = ch.dropna(subset=[CLIENT_ID_COL, "DATE_EVENT", PRODUCT_COL])
        if ch.empty:
            continue

        ch["PARTITION_MONTH"] = ch["DATE_EVENT"].dt.to_period("M").astype(str)
        ch = ch[ch["PARTITION_MONTH"] >= min_month_inclusive]

        if exclude_class:
            ch = ch[ch[PRODUCT_COL].astype(str).str.strip() != "Aucune_Proposition"]

        if ch.empty:
            continue

        # rename & keep
        keep_cols = {
            CLIENT_ID_COL: "CLIENT_ID",
            "DATE_EVENT": "DATE_EVENT",
            PRODUCT_COL: "PRODUCT_CODE",
            "PARTITION_MONTH": "PARTITION_MONTH"
        }
        ch = ch.rename(columns=keep_cols)[list(keep_cols.values()) + list(EXTRA_EVENT_COLS)]
        out.append(ch)
        total += len(ch)

        if total >= limit_rows:
            break

    if not out:
        return pd.DataFrame(columns=["CLIENT_ID", "DATE_EVENT", "PRODUCT_CODE", "PARTITION_MONTH"] + list(EXTRA_EVENT_COLS))

    df = pd.concat(out, ignore_index=True)
    # dédup stricte + head LIMIT_ROWS
    df = (df.sort_values(["CLIENT_ID", "DATE_EVENT"])
            .drop_duplicates(subset=["CLIENT_ID", "DATE_EVENT"], keep="last")
            .head(limit_rows)
            .reset_index(drop=True))
    return df


def main():
    np.random.seed(RANDOM_SEED)
    log.info("=== RECIPE 1 FAST START ===")
    ds_in = dataiku.Dataset(DATASET_MAIN)
    ds_out = dataiku.Dataset(OUTPUT_DATASET_NAME)

    parts = _list_partitions_if_any(ds_in)
    if parts:
        # Lecture directe des dernières partitions seulement
        target_parts = parts[:max(1, MONTHS_BACK_TARGET)]
        log.info(f"Partitions détectées → on lit: {target_parts}")
        cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)
        chunks = []
        for p in target_parts:
            try:
                ch = ds_in.get_dataframe(partitions=p, columns=cols, limit=None,
                                         parse_dates=False, infer_with_pandas=True)
                if ch.empty:
                    continue
                ch["DATE_EVENT"] = pd.to_datetime(ch[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
                ch = ch.dropna(subset=[CLIENT_ID_COL, "DATE_EVENT", PRODUCT_COL])
                if EXCLUDE_ON_EVENTS:
                    ch = ch[ch[PRODUCT_COL].astype(str).str.strip() != "Aucune_Proposition"]
                if ch.empty:
                    continue
                ch["PARTITION_MONTH"] = p if isinstance(p, str) else str(p)
                ch = ch.rename(columns={CLIENT_ID_COL:"CLIENT_ID", PRODUCT_COL:"PRODUCT_CODE"})
                chunks.append(ch[["CLIENT_ID","DATE_EVENT","PRODUCT_CODE","PARTITION_MONTH"] + list(EXTRA_EVENT_COLS)])
            except Exception:
                continue
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["CLIENT_ID","DATE_EVENT","PRODUCT_CODE","PARTITION_MONTH"]+list(EXTRA_EVENT_COLS))
        if df.empty:
            raise RuntimeError("Partitions trouvées mais aucun enregistrement lisible.")

        # équilibre simple et tronque
        df["_m"] = df["PARTITION_MONTH"]
        per_m = max(1, int(np.ceil(LIMIT_ROWS / max(1, df["_m"].nunique()))))
        df = (df.groupby("_m", group_keys=False).apply(lambda g: g.head(per_m))).drop(columns=["_m"])
        df = (df.sort_values(["CLIENT_ID","DATE_EVENT"])
                .drop_duplicates(subset=["CLIENT_ID","DATE_EVENT"], keep="last")
                .head(LIMIT_ROWS)
                .reset_index(drop=True))
    else:
        # Pas de partitions => 2 passes max (mois max, puis stream filtré)
        max_month = _find_max_month_by_scan(ds_in)
        if not max_month:
            raise RuntimeError("Impossible de déterminer le mois max (TIME_COL vide ?).")
        start_month = _month_minus(max_month, MONTHS_BACK_TARGET-1)
        log.info(f"Aucune partition. max_month={max_month} | start_month={start_month}")
        df = _stream_filter_last_months(ds_in, start_month, LIMIT_ROWS, EXCLUDE_ON_EVENTS)

    # Stats & écritures
    if df.empty:
        raise RuntimeError("Échantillon vide (même après fast path).")
    n_clients = df["CLIENT_ID"].nunique()
    n_months  = df["PARTITION_MONTH"].nunique()
    log.info(f"FAST sample -> rows={len(df):,} | clients={n_clients:,} | months={n_months}")

    ds_out.write_with_schema(df)
    log.info(f" Wrote dataset '{OUTPUT_DATASET_NAME}' with {len(df):,} rows.")
    log.info("=== RECIPE 1 FAST DONE ===")


if __name__ == "__main__":
    main()

