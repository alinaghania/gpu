"""
RECIPE 1 — EVENTS SAMPLED (fast & tiny)
----------------------------------------------
Lit un petit échantillon d'événements depuis BASE_SCORE_COMPLETE_prepared
sur les derniers mois (3 par défaut), exclut "Aucune_Proposition" (optionnel),
déduplique (CLIENT_ID, DATE_EVENT), et écrit un dataset prêt pour le SequenceBuilder.

Sortie: dataset "events_sampled"
Schema (inféré) :
- CLIENT_ID        (string ou bigint selon tes données)
- DATE_EVENT       (timestamp)
- PRODUCT_CODE     (string)
- PARTITION_MONTH  (string, format "YYYY-MM")

NOTE: paramétrage mini pour lancer un smoke-test end-to-end rapidement.
"""

import os, sys, logging
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import dataiku

# ===================== PARAMS UTILISATEUR =========================
DATASET_MAIN          = "BASE_SCORE_COMPLETE_prepared"   # input
OUTPUT_DATASET_NAME   = "events_sampled"                 # output

# Colonnes source (doivent matcher la base)
CLIENT_ID_COL         = "NUMTECPRS"
TIME_COL              = "DATMAJ"                         # date/mois
PRODUCT_COL           = "SOUSCRIPTION_PRODUIT_1M"
EXTRA_EVENT_COLS      = []                               # ex: ["CANAL", "FAMILLE"]

# Échantillonnage ultra-light
LIMIT_ROWS            = 400          # <= très petit pour un run rapide
MONTHS_BACK_TARGET    = 3            # derniers N mois à viser
PER_MONTH_CAP         = None         # auto (LIMIT_ROWS // MONTHS_BACK_TARGET)
MIN_CLIENT_MONTHS     = 1            # tolérant : au moins 1 pas de temps
EXCLUDE_ON_EVENTS     = True         # exclure "Aucune_Proposition"
RANDOM_SEED           = 42

# Lecture en chunks (utile si dataset non partitionné)
READ_IN_CHUNKS        = True
CHUNKSIZE             = 5_000

# ===================== LOGGING =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("recipe1")


# ===================== HELPERS =========================
def _to_month_label(p) -> str:
    try:
        return pd.to_datetime(p).strftime("%Y-%m")
    except Exception:
        s = str(p)
        if len(s) == 7 and s[4] == "-":
            return s
        if len(s) in (6, 8) and s[:4].isdigit():
            return f"{s[:4]}-{s[4:6]}"
        return s


def _list_month_partitions(ds: dataiku.Dataset) -> List[str]:
    """Essaie d'abord les partitions DSS, sinon infère via la colonne date."""
    # 1) Partitions DSS
    try:
        parts = ds.list_partitions()
        parts = sorted(parts, reverse=True)
        if parts:
            return [_to_month_label(x) for x in parts]
    except Exception:
        pass

    # 2) Fallback: inférence via DATMAJ
    months = set()
    cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL]
    for ch in ds.iter_dataframes(columns=cols, chunksize=CHUNKSIZE,
                                 parse_dates=False, infer_with_pandas=True):
        dt = pd.to_datetime(ch[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
        months.update(dt.dropna().dt.to_period("M").astype(str).unique().tolist())
        if len(months) >= 48:  # on limite la découverte
            break
    return sorted(list(months), reverse=True)


def _read_month_sample(
    ds: dataiku.Dataset,
    month_label: str,
    per_month_cap: Optional[int],
    exclude_on_events: bool
) -> pd.DataFrame:
    """Lit un mois (partition ou filtre DATMAJ) et fait le nettoyage minimal."""
    cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)

    # 1) Tentative partitions
    try:
        df = ds.get_dataframe(partitions=month_label, columns=cols,
                              limit=None, parse_dates=False, infer_with_pandas=True)
    except Exception:
        # 2) Fallback: filtre DATMAJ = month_label en chunks
        dfs = []
        for ch in ds.iter_dataframes(columns=cols, chunksize=CHUNKSIZE,
                                     parse_dates=False, infer_with_pandas=True):
            dt = pd.to_datetime(ch[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
            ch = ch.loc[dt.dt.to_period("M").astype(str) == month_label, cols]
            if not ch.empty:
                dfs.append(ch)
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=cols)

    if df.empty:
        return df

    # Parse & clean
    df = df.rename(columns={
        CLIENT_ID_COL: "CLIENT_ID",
        TIME_COL: "DATE_EVENT",
        PRODUCT_COL: "PRODUCT_CODE"
    })
    df["DATE_EVENT"] = pd.to_datetime(df["DATE_EVENT"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["CLIENT_ID", "DATE_EVENT"])

    if exclude_on_events:
        df = df[df["PRODUCT_CODE"].astype(str).strip() != "Aucune_Proposition"]

    if df.empty:
        return df

    # Dédup stricte (CLIENT_ID, DATE_EVENT)
    df = (df.sort_values(["CLIENT_ID", "DATE_EVENT"])
            .drop_duplicates(subset=["CLIENT_ID", "DATE_EVENT"], keep="last"))

    # Cap par mois
    if per_month_cap and len(df) > per_month_cap:
        df = df.sample(n=per_month_cap, random_state=RANDOM_SEED)

    return df


def build_events_stratified(
    limit_rows: int,
    months_back: int,
    per_month_cap: Optional[int],
    min_client_months: int,
    exclude_on_events: bool
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ds = dataiku.Dataset(DATASET_MAIN)
    months_all = _list_month_partitions(ds)
    if not months_all:
        raise RuntimeError("Impossible de déterminer les mois disponibles (partitions/DATMAJ).")

    target_months = months_all[:max(1, months_back)]
    if not per_month_cap or per_month_cap <= 0:
        per_month_cap = max(50, limit_rows // max(1, len(target_months)))

    log.info(f"[STEP1] limit_rows={limit_rows:,} | keep_months={months_back} | per_month_cap={per_month_cap} | exclude={exclude_on_events}")

    # Lecture stratifiée par mois
    chunks = []
    total = 0
    for m in target_months:
        dfm = _read_month_sample(ds, m, per_month_cap=per_month_cap, exclude_on_events=exclude_on_events)
        if not dfm.empty:
            dfm["PARTITION_MONTH"] = m
            chunks.append(dfm)
            total += len(dfm)
        if total >= limit_rows * 2:
            break

    base_cols = ["CLIENT_ID", "DATE_EVENT", "PRODUCT_CODE", "PARTITION_MONTH"] + list(EXTRA_EVENT_COLS)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=base_cols)

    # Fallbacks si trop vide
    if df.empty and exclude_on_events:
        log.warning("Vide après exclusion — retry sans exclusion sur les mêmes mois.")
        chunks = []
        for m in target_months:
            dfm = _read_month_sample(ds, m, per_month_cap=per_month_cap, exclude_on_events=False)
            if not dfm.empty:
                dfm["PARTITION_MONTH"] = m
                chunks.append(dfm)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=base_cols)

    if df.empty:
        raise RuntimeError("Aucun événement trouvé après fallback.")

    # Profondeur min par client (très permissif ici)
    tmp = df.copy()
    tmp["_m"] = tmp["DATE_EVENT"].dt.to_period("M")
    keep_ids = tmp.groupby("CLIENT_ID")["_m"].nunique()
    keep_ids = set(keep_ids[keep_ids >= min_client_months].index)
    df_kept = tmp[tmp["CLIENT_ID"].isin(keep_ids)].drop(columns=["_m"])
    if df_kept.empty:
        df_kept = tmp.drop(columns=["_m"])

    # Équilibrage par mois + tronque à LIMIT_ROWS
    df_kept["_month"] = df_kept["DATE_EVENT"].dt.to_period("M").astype(str)
    n_target_months = df_kept["_month"].nunique()
    per_m = max(1, int(np.ceil(limit_rows / max(1, n_target_months))))
    df_kept = (df_kept.groupby("_month", group_keys=False).apply(lambda g: g.head(per_m)))
    df_kept = df_kept.drop(columns=["_month"]).head(limit_rows).reset_index(drop=True)

    # Stats
    n_clients = df_kept["CLIENT_ID"].nunique()
    n_months  = df_kept["DATE_EVENT"].dt.to_period("M").nunique()
    meta = {
        "selected_months": target_months,
        "per_month_cap": per_month_cap,
        "exclude_on_events": exclude_on_events,
        "n_clients": int(n_clients),
        "n_months": int(n_months),
        "rows": int(len(df_kept)),
    }
    return df_kept, meta


# ===================== MAIN (RECIPE) =========================
def main():
    np.random.seed(RANDOM_SEED)

    log.info("=== RECIPE 1 START (events sampled) ===")
    ds_out = dataiku.Dataset(OUTPUT_DATASET_NAME)

    df_events, meta = build_events_stratified(
        limit_rows=LIMIT_ROWS,
        months_back=MONTHS_BACK_TARGET,
        per_month_cap=PER_MONTH_CAP,
        min_client_months=MIN_CLIENT_MONTHS,
        exclude_on_events=EXCLUDE_ON_EVENTS
    )

    log.info(f"Rows={meta['rows']:,} | Clients={meta['n_clients']:,} | Months={meta['n_months']}")
    try:
        vc = df_events["PRODUCT_CODE"].astype(str).value_counts().head(10)
        log.info(f"Top target values:\n{vc}")
    except Exception:
        pass

    # Écriture (schema auto)
    ds_out.write_with_schema(df_events)
    log.info(f" Wrote dataset '{OUTPUT_DATASET_NAME}' with {len(df_events):,} rows.")
    log.info("=== RECIPE 1 DONE ===")


if __name__ == "__main__":
    main()

