"""
RECIPE 1 — EVENTS SAMPLED (FAST & MAX LOGS)
----------------------------------------------
But: construire un tout petit échantillon récent pour smoke-test end-to-end,
avec des logs détaillés (étapes, timings, compteurs, mémoire, partitions testées).

Entrée : BASE_SCORE_COMPLETE_prepared
Sortie : events_sampled

Colonnes de sortie :
- CLIENT_ID
- DATE_EVENT (timestamp naïf, timezone-stripped)
- PRODUCT_CODE
- PARTITION_MONTH (YYYY-MM)
"""

import sys, os, time, gc, logging
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import dataiku

# ============== CONFIGS ==============
DATASET_MAIN        = "BASE_SCORE_COMPLETE_prepared"
OUTPUT_DATASET_NAME = "events_sampled"

# Colonnes source
CLIENT_ID_COL    = "NUMTECPRS"
TIME_COL         = "DATMAJ"
PRODUCT_COL      = "SOUSCRIPTION_PRODUIT_1M"
EXTRA_EVENT_COLS = []  # ex: ["CANAL", "FAMILLE"]

# Échantillon minuscule et récent (smoke-test)
LIMIT_ROWS         = 400
MONTHS_BACK_TARGET = 2
MIN_CLIENT_MONTHS  = 1
EXCLUDE_ON_EVENTS  = False  # laisse à False pour la vitesse

# Lecture streaming
CHUNKSIZE          = 200_000
RANDOM_SEED        = 42

# Logs
LOG_LEVEL          = logging.DEBUG  # DEBUG pour tout voir, INFO pour normal
LOG_MEMORY         = True           # log RSS mémoire si psutil dispo
LOG_CHUNK_EVERY_N  = 1              # log chaque chunk (1) ; augmente si trop bavard

# ============== LOGGING ==============
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("recipe1_fast_verbose")

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

def _mem_tag(tag: str):
    if LOG_MEMORY and _HAS_PSUTIL:
        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        log.debug(f"[MEM] {tag}: RSS≈{rss:.1f} MB")

def _timeit_start(label: str) -> float:
    log.debug(f"[TIMER] START: {label}")
    return time.time()

def _timeit_end(label: str, t0: float):
    dt = time.time() - t0
    log.debug(f"[TIMER] END  : {label} | {dt:.3f}s")

def _df_quick_stats(df: pd.DataFrame, name: str, max_show: int = 5):
    try:
        log.debug(f"[DF] {name}: shape={df.shape} | cols={list(df.columns)}")
        if len(df) > 0:
            log.debug(f"[DF] {name}: head={df.head(max_show).to_dict(orient='records')}")
            if "PRODUCT_CODE" in df.columns:
                vc = df["PRODUCT_CODE"].astype(str).value_counts().head(max_show)
                log.debug(f"[DF] {name}: top PRODUCT_CODE ->\n{vc}")
    except Exception as e:
        log.debug(f"[DF] {name}: quick_stats error: {e}")

# ============== HELPERS ==============
def _to_month_label(dt_series: pd.Series) -> pd.Series:
    return pd.to_datetime(dt_series, errors="coerce", utc=True).dt.tz_convert(None).dt.to_period("M").astype(str)

def _list_partitions_if_any(ds: dataiku.Dataset) -> List[str]:
    t0 = _timeit_start("list_partitions")
    try:
        parts = ds.list_partitions()
        parts = sorted(parts, reverse=True)
        log.info(f"[PART] list_partitions -> {len(parts)} partitions (show up to 10): {parts[:10]}")
        return parts
    except Exception as e:
        log.info(f"[PART] list_partitions not available ({e}). Will stream.")
        return []
    finally:
        _timeit_end("list_partitions", t0)

def _find_max_month_by_scan(ds: dataiku.Dataset) -> Optional[str]:
    """1 pass légère (TIME_COL only) pour trouver le mois max si pas de partitions."""
    t0 = _timeit_start("scan_max_month")
    max_month = None
    chunks = 0
    for ch in ds.iter_dataframes(columns=[TIME_COL], chunksize=CHUNKSIZE,
                                 parse_dates=False, infer_with_pandas=True):
        chunks += 1
        if chunks % LOG_CHUNK_EVERY_N == 0:
            log.debug(f"[SCAN] chunk#{chunks} read for max-month | ch.shape={ch.shape}")
            _mem_tag(f"scan_max_month chunk#{chunks}")
        m = _to_month_label(ch[TIME_COL]).dropna()
        if not m.empty:
            mm = m.max()
            if (max_month is None) or (mm > max_month):
                log.debug(f"[SCAN] new max_month={mm}")
                max_month = mm
        # option: early stop si on a trouvé un mois “proche” de now (décommenter si utile)
        # if max_month and chunks >= 3: break
    _timeit_end("scan_max_month", t0)
    log.info(f"[SCAN] max_month={max_month} after {chunks} chunks")
    return max_month

def _month_minus(month_str: str, k: int) -> str:
    p = pd.Period(month_str, freq="M")
    return (p - k).strftime("%Y-%m")

def _try_read_partition(ds: dataiku.Dataset, p: str) -> Dict[str, pd.DataFrame]:
    """
    Essaie plusieurs formats de spec partitions:
      - "2025-05"
      - "DATMAJ=2025-05"
      - "month=2025-05"
      - "PARTITION_MONTH=2025-05"
    Retourne dict {spec_str: df}
    """
    specs = [p, f"{TIME_COL}={p}", f"month={p}", f"PARTITION_MONTH={p}"]
    out = {}
    for spec in specs:
        t0 = _timeit_start(f"read_partitions[{spec}]")
        try:
            df = ds.get_dataframe(partitions=spec, limit=None, parse_dates=False, infer_with_pandas=True)
            out[spec] = df
            log.debug(f"[PART-TRY] spec={spec} -> shape={df.shape}")
        except Exception as e:
            log.debug(f"[PART-TRY] spec={spec} -> ERROR {e}")
        finally:
            _timeit_end(f"read_partitions[{spec}]", t0)
            _mem_tag(f"read_partitions[{spec}]")
    return out

def _stream_filter_last_months(ds: dataiku.Dataset, min_month_inclusive: str,
                               limit_rows: int, exclude_class: bool) -> pd.DataFrame:
    """2e pass : filtre en streaming sur les derniers mois, stop early dès LIMIT_ROWS atteint."""
    t0 = _timeit_start("stream_filter_last_months")
    cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)
    out = []
    total = 0
    chunks = 0

    for ch in ds.iter_dataframes(columns=cols, chunksize=CHUNKSIZE,
                                 parse_dates=False, infer_with_pandas=True):
        chunks += 1
        if chunks % LOG_CHUNK_EVERY_N == 0:
            log.debug(f"[STREAM] chunk#{chunks} raw shape={ch.shape} (cum kept={total})")
        _mem_tag(f"stream chunk#{chunks} [before parse]")

        # parse date & month
        ch["DATE_EVENT"] = pd.to_datetime(ch[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
        before_drop = len(ch)
        ch = ch.dropna(subset=[CLIENT_ID_COL, "DATE_EVENT", PRODUCT_COL])
        after_drop = len(ch)
        if chunks % LOG_CHUNK_EVERY_N == 0:
            log.debug(f"[STREAM] chunk#{chunks} dropna rows: {before_drop} -> {after_drop}")

        if ch.empty:
            continue

        ch["PARTITION_MONTH"] = ch["DATE_EVENT"].dt.to_period("M").astype(str)
        ch = ch[ch["PARTITION_MONTH"] >= min_month_inclusive]

        if exclude_class:
            # attention : str accessor
            ch = ch[ch[PRODUCT_COL].astype(str).str.strip() != "Aucune_Proposition"]

        if ch.empty:
            continue

        # rename & keep
        ch = ch.rename(columns={CLIENT_ID_COL: "CLIENT_ID", PRODUCT_COL: "PRODUCT_CODE"})
        keep = ch[["CLIENT_ID", "DATE_EVENT", "PRODUCT_CODE", "PARTITION_MONTH"] + list(EXTRA_EVENT_COLS)]
        out.append(keep)
        total += len(keep)

        if chunks % LOG_CHUNK_EVERY_N == 0:
            log.debug(f"[STREAM] chunk#{chunks} kept={len(keep)} | cum={total} | head:\n{keep.head(3)}")

        _mem_tag(f"stream chunk#{chunks} [after filter]")
        if total >= limit_rows:
            log.info(f"[STREAM] early-stop reached LIMIT_ROWS={limit_rows} at chunk#{chunks}")
            break

    if not out:
        _timeit_end("stream_filter_last_months", t0)
        log.warning("[STREAM] no rows collected in streaming.")
        return pd.DataFrame(columns=["CLIENT_ID", "DATE_EVENT", "PRODUCT_CODE", "PARTITION_MONTH"] + list(EXTRA_EVENT_COLS))

    df = pd.concat(out, ignore_index=True)
    log.debug(f"[STREAM] concat result shape={df.shape} before dedup/head")
    df = (df.sort_values(["CLIENT_ID", "DATE_EVENT"])
            .drop_duplicates(subset=["CLIENT_ID", "DATE_EVENT"], keep="last")
            .head(limit_rows)
            .reset_index(drop=True))
    _timeit_end("stream_filter_last_months", t0)
    _df_quick_stats(df, "STREAM_RESULT")
    return df

# ============== MAIN ==============
def main():
    np.random.seed(RANDOM_SEED)
    log.info("=== RECIPE 1 FAST & VERBOSE START ===")
    _mem_tag("boot")

    t_init = time.time()
    ds_in = dataiku.Dataset(DATASET_MAIN)
    ds_out = dataiku.Dataset(OUTPUT_DATASET_NAME)

    # 1) Partitions ?
    parts = _list_partitions_if_any(ds_in)
    if parts:
        target_parts = parts[:max(1, MONTHS_BACK_TARGET)]
        log.info(f"[PART] target_parts={target_parts} (MONTHS_BACK_TARGET={MONTHS_BACK_TARGET})")

        chunks = []
        total_rows = 0
        for p in target_parts:
            log.info(f"[PART] === reading partition '{p}' ===")
            tried = _try_read_partition(ds_in, p)

            # choisi le 1er spec non vide
            usable_spec = None
            dfp = None
            for spec, df in tried.items():
                if df is not None and len(df) > 0:
                    usable_spec = spec
                    dfp = df
                    break

            if dfp is None or dfp.empty:
                log.warning(f"[PART] partition '{p}': all specs empty -> skip")
                continue

            _df_quick_stats(dfp, f"PART_RAW[{usable_spec}]")

            # Vérifie colonnes requises
            missing = [c for c in [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] if c not in dfp.columns]
            if missing:
                log.warning(f"[PART] missing required columns {missing} in spec={usable_spec} -> skip")
                continue

            # Parse & filtre
            t0 = _timeit_start(f"parse_filter[{p}]")
            dfp["DATE_EVENT"] = pd.to_datetime(dfp[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
            before_drop = len(dfp)
            dfp = dfp.dropna(subset=[CLIENT_ID_COL, "DATE_EVENT", PRODUCT_COL])
            after_drop = len(dfp)
            log.debug(f"[PART] [{p}] dropna rows: {before_drop} -> {after_drop}")

            if EXCLUDE_ON_EVENTS:
                dfp = dfp[dfp[PRODUCT_COL].astype(str).str.strip() != "Aucune_Proposition"]

            if dfp.empty:
                log.warning(f"[PART] [{p}] empty after parse/filter -> skip")
                _timeit_end(f"parse_filter[{p}]", t0)
                continue

            # Ajoute PARTITION_MONTH depuis la date (plus fiable que p si format exotique)
            dfp["PARTITION_MONTH"] = dfp["DATE_EVENT"].dt.to_period("M").astype(str)
            dfp = dfp.rename(columns={CLIENT_ID_COL: "CLIENT_ID", PRODUCT_COL: "PRODUCT_CODE"})
            keep = dfp[["CLIENT_ID", "DATE_EVENT", "PRODUCT_CODE", "PARTITION_MONTH"] + list(EXTRA_EVENT_COLS)]
            chunks.append(keep)
            total_rows += len(keep)
            _timeit_end(f"parse_filter[{p}]", t0)

            log.info(f"[PART] [{p}] usable_spec={usable_spec} -> kept={len(keep)} | cum={total_rows}")
            _mem_tag(f"part[{p}] kept")

            # petit early-stop global si on a très largement de quoi faire
            if total_rows >= LIMIT_ROWS * 4:
                log.info(f"[PART] early-break partitions loop (got {total_rows} rows)")
                break

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["CLIENT_ID","DATE_EVENT","PRODUCT_CODE","PARTITION_MONTH"]+list(EXTRA_EVENT_COLS))

        if df.empty:
            log.error("[PART] Detected partitions but nothing usable — fallback to streaming scan.")
            # Fallback global: 2 passes (mois max, puis streaming filtré)
            max_month = _find_max_month_by_scan(ds_in)
            if not max_month:
                raise RuntimeError("Impossible de déterminer le mois max (TIME_COL vide ?).")
            start_month = _month_minus(max_month, MONTHS_BACK_TARGET-1)
            log.info(f"[FALLBACK] streaming. max_month={max_month} | start_month={start_month}")
            df = _stream_filter_last_months(ds_in, start_month, LIMIT_ROWS, EXCLUDE_ON_EVENTS)
        else:
            # Équilibrage simple par mois + tronque
            log.debug(f"[PART] concat result shape={df.shape} before balance/truncate")
            df["_m"] = df["PARTITION_MONTH"]
            per_m = max(1, int(np.ceil(LIMIT_ROWS / max(1, df["_m"].nunique()))))
            log.debug(f"[PART] per_m={per_m} for balancing over {_safe_nunique(df, '_m')} months")
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
        log.info(f"[NOPART] max_month={max_month} | start_month={start_month}")
        df = _stream_filter_last_months(ds_in, start_month, LIMIT_ROWS, EXCLUDE_ON_EVENTS)

    # 2) Stats & écriture
    if df.empty:
        raise RuntimeError("Échantillon vide (même après fast path).")
    n_clients = df["CLIENT_ID"].nunique()
    n_months  = df["PARTITION_MONTH"].nunique()
    log.info(f"[RESULT] rows={len(df):,} | clients={n_clients:,} | months={n_months}")
    _df_quick_stats(df, "FINAL_SAMPLE")
    _mem_tag("before_write")

    ds_out.write_with_schema(df)
    log.info(f"[WRITE] dataset '{OUTPUT_DATASET_NAME}' written with {len(df):,} rows.")

    total_time = time.time() - t_init
    log.info(f"=== RECIPE 1 FAST & VERBOSE DONE in {total_time:.2f}s ===")
    _mem_tag("end")
    gc.collect()

def _safe_nunique(df: pd.DataFrame, col: str) -> int:
    try:
        return df[col].nunique()
    except Exception:
        return -1

if __name__ == "__main__":
    main()



