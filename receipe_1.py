"""
RECIPE 1 — EVENTS SAMPLED (FAST & ROBUST)
----------------------------------------------
But: construire un tout petit échantillon récent pour smoke-test end-to-end,
en maximisant la vitesse (pas de rescans par mois, stop early dès quota atteint),
et en étant TOLÉRANT aux specs de partitions Dataiku.

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
EXCLUDE_ON_EVENTS  = False        # pas d'exclusion pour aller VITE

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


def _try_read_partition(ds: dataiku.Dataset, p: str, need_cols: List[str]) -> pd.DataFrame:
    """
    Essaie plusieurs formats de spec partitions:
      - "2025-05"
      - "DATMAJ=2025-05"
      - "month=2025-05"
      - "PARTITION_MONTH=2025-05"
    Retourne un DF (possiblement vide). Ne lève pas.
    """
    candidates = [p, f"{TIME_COL}={p}", f"month={p}", f"PARTITION_MONTH={p}"]
    for spec in candidates:
        try:
            df = ds.get_dataframe(partitions=spec, limit=None, parse_dates=False, infer_with_pandas=True)
            if df is not None and len(df) > 0:
                df["_PART_SPEC_USED_"] = spec
                # si les colonnes demandées n'existent pas, on lira toutes puis on filtrera après
                missing = [c for c in need_cols if c not in df.columns]
                if missing:
                    # relit sans restriction de colonnes (certaines FS ne gèrent pas "columns=" avec partitions)
                    df = ds.get_dataframe(partitions=spec, limit=None, parse_dates=False, infer_with_pandas=True)
                    df["_PART_SPEC_USED_"] = spec
                return df
        except Exception:
            continue
    return pd.DataFrame()


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
        ch = ch.rename(columns={CLIENT_ID_COL: "CLIENT_ID", PRODUCT_COL: "PRODUCT_CODE"})
        out.append(ch[["CLIENT_ID", "DATE_EVENT", "PRODUCT_CODE", "PARTITION_MONTH"] + list(EXTRA_EVENT_COLS)])
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
        target_parts = parts[:max(1, MONTHS_BACK_TARGET)]
        log.info(f"Partitions détectées → on lit: {target_parts}")

        need_cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)
        chunks = []
        for p in target_parts:
            dfp = _try_read_partition(ds_in, p, need_cols)
            if dfp.empty:
                log.warning(f"Partition '{p}' lue vide (ou spec incompatible). On passe.")
                continue

            # harmonise colonnes
            if CLIENT_ID_COL not in dfp.columns or TIME_COL not in dfp.columns or PRODUCT_COL not in dfp.columns:
                # on tente des alias fréquents (si besoin tu peux en ajouter)
                alias_map = {}
                # rien par défaut ; on s'appuie sur TIME_COL/CLIENT_ID_COL/PRODUCT_COL fournis
                # -> si absent, on abandonne cette partition
                missing = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL]
                has_all = all(c in dfp.columns for c in missing)
                if not has_all:
                    log.warning(f"Colonnes requises absentes dans partition '{p}' (spec {dfp.get('_PART_SPEC_USED_', ['?'])[0]}). Skip.")
                    continue

            # parse / filtre
            dfp["DATE_EVENT"] = pd.to_datetime(dfp[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
            dfp = dfp.dropna(subset=[CLIENT_ID_COL, "DATE_EVENT", PRODUCT_COL])
            if dfp.empty:
                continue

            if EXCLUDE_ON_EVENTS:
                dfp = dfp[dfp[PRODUCT_COL].astype(str).str.strip() != "Aucune_Proposition"]

            if dfp.empty:
                continue

            dfp["PARTITION_MONTH"] = dfp["DATE_EVENT"].dt.to_period("M").astype(str)
            dfp = dfp.rename(columns={CLIENT_ID_COL: "CLIENT_ID", PRODUCT_COL: "PRODUCT_CODE"})
            chunks.append(dfp[["CLIENT_ID", "DATE_EVENT", "PRODUCT_CODE", "PARTITION_MONTH"] + list(EXTRA_EVENT_COLS)])

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["CLIENT_ID","DATE_EVENT","PRODUCT_CODE","PARTITION_MONTH"]+list(EXTRA_EVENT_COLS))

        if df.empty:
            log.error("Partitions détectées mais rien de lisible — on bascule en mode scan rapide (fallback).")
            # Fallback global: 2 passes (mois max, puis streaming filtré)
            max_month = _find_max_month_by_scan(ds_in)
            if not max_month:
                raise RuntimeError("Impossible de déterminer le mois max (TIME_COL vide ?).")
            start_month = _month_minus(max_month, MONTHS_BACK_TARGET-1)
            log.info(f"Fallback streaming. max_month={max_month} | start_month={start_month}")
            df = _stream_filter_last_months(ds_in, start_month, LIMIT_ROWS, EXCLUDE_ON_EVENTS)
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


