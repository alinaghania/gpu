"""
RECIPE 2 — TRAINING (GPU-friendly, tiny)
----------------------------------
Lit 'events_sampled', entraîne le modèle hybride T4Rec+Transformer,
et écrit 3 sorties :
  - T4REC_METRICS
  - T4REC_PREDICTIONS
  - T4REC_MODEL_ARTIFACTS

Paramétrage mini pour un run rapide (1-2 époques, petit d_model), avec GPU/AMP via pipeline_core.
"""

import os, sys, logging, time, json
from datetime import datetime

import numpy as np
import pandas as pd
import dataiku

from t4rec_toolkit.pipeline_core import run_training, blank_config

# ===================== PARAMS =====================
INPUT_EVENTS_DATASET      = "events_sampled"

OUTPUT_METRICS_DATASET    = "T4REC_METRICS"
OUTPUT_PREDICTIONS_DATASET= "T4REC_PREDICTIONS"
OUTPUT_ARTIFACTS_DATASET  = "T4REC_MODEL_ARTIFACTS"

# Colonnes (doivent matcher Recipe 1)
CLIENT_ID_COL  = "CLIENT_ID"
TIME_COL       = "DATE_EVENT"
PRODUCT_COL    = "PRODUCT_CODE"
EXTRA_EVENT_COLS = []  # si ajouté en Recipe 1, liste-les ici aussi

# Hyperparams tiny + GPU friendly
D_MODEL    = 128
N_HEADS    = 2
N_LAYERS   = 2
BATCH_SIZE = 32            # petit batch, AMP activé => rapide sur GPU
EPOCHS     = 2             # 1-2 suffisent pour smoke-test
LEARNING_RATE = 5e-4
VAL_SPLIT     = 0.20

# Fenêtre temporelle souhaitée (borne par les mois réellement présents)
WISHED_LOOKBACK = 6        # très court pour run rapide

# Classes rares (fusion agressive => stabilité sur tiny sample)
MERGE_RARE_THRESHOLD = 200
OTHER_CLASS_NAME     = "AUTRES_PRODUITS"

RANDOM_SEED = 42

# (Option) Limite côté pipeline pour des runs-éclair, désactive en prod
DEBUG_LIMIT_CLIENTS = 500   # ~quelques centaines d’exemples


# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("recipe2")


# ===================== MAIN =====================
def main():
    np.random.seed(RANDOM_SEED)

    log.info("=== RECIPE 2 START (training tiny GPU) ===")
    ds_in = dataiku.Dataset(INPUT_EVENTS_DATASET)

    # 1) Lire les événements (tout le dataset ; déjà échantillonné en Recipe 1)
    cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)
    df_events = ds_in.get_dataframe(columns=cols, parse_dates=False, infer_with_pandas=True)
    # Parse date proprement
    df_events[TIME_COL] = pd.to_datetime(df_events[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
    df_events = df_events.dropna(subset=[CLIENT_ID_COL, TIME_COL, PRODUCT_COL])

    if df_events.empty:
        raise RuntimeError("Input events dataset is empty.")

    # Quelques stats simples
    n_clients = df_events[CLIENT_ID_COL].nunique()
    n_months  = df_events[TIME_COL].dt.to_period("M").nunique()
    vc = df_events[PRODUCT_COL].astype(str).value_counts().head(10)
    log.info(f"Events read: rows={len(df_events):,} | clients={n_clients:,} | months={n_months}")
    log.info(f"Top labels:\n{vc}")

    # 2) Config pipeline
    cfg = blank_config()

    # Données (passage en mémoire → run_training ne relira pas Dataiku)
    cfg["data"]["events_df"]        = df_events
    cfg["data"]["events_dataset"]   = ""  # ignoré car events_df fourni
    cfg["data"]["client_id_col"]    = CLIENT_ID_COL
    cfg["data"]["event_time_col"]   = TIME_COL
    cfg["data"]["product_col"]      = PRODUCT_COL
    cfg["data"]["event_extra_cols"] = list(EXTRA_EVENT_COLS)

    # Profil OFF (mémoire)
    cfg["data"]["dataset_name"]             = ""
    cfg["data"]["profile_categorical_cols"] = []
    cfg["data"]["profile_sequence_cols"]    = []
    cfg["data"]["profile_df"]               = None
    cfg["data"]["profile_join_key"]         = CLIENT_ID_COL

    # Séquence — borne par les mois réellement présents
    months_present = int(n_months)
    lookback = max(1, min(WISHED_LOOKBACK, months_present))
    cfg["sequence"]["months_lookback"]          = lookback
    cfg["sequence"]["time_granularity"]         = "M"
    cfg["sequence"]["min_events_per_client"]    = 1
    cfg["sequence"]["target_horizon"]           = 1
    cfg["sequence"]["pad_value"]                = 0
    cfg["sequence"]["build_target_from_events"] = True

    #  On n’exclut pas ici (déjà géré côté events en Recipe 1)
    cfg["features"]["exclude_target_values"] = []
    cfg["features"]["merge_rare_threshold"]  = MERGE_RARE_THRESHOLD
    cfg["features"]["other_class_name"]      = OTHER_CLASS_NAME

    # Modèle (petit) GPU friendly
    cfg["model"]["d_model"]             = D_MODEL
    cfg["model"]["n_heads"]             = N_HEADS
    cfg["model"]["n_layers"]            = N_LAYERS
    cfg["model"]["dropout"]             = 0.10
    cfg["model"]["max_sequence_length"] = lookback
    cfg["model"]["vocab_size"]          = 2000

    # Entraînement (GPU)
    cfg["training"]["batch_size"]      = BATCH_SIZE
    cfg["training"]["num_epochs"]      = EPOCHS
    cfg["training"]["learning_rate"]   = LEARNING_RATE
    cfg["training"]["weight_decay"]    = 1e-4
    cfg["training"]["val_split"]       = VAL_SPLIT
    cfg["training"]["class_weighting"] = True
    cfg["training"]["gradient_clip"]   = 1.0
    cfg["training"]["optimizer"]       = "adamw"

    # >>> GPU/AMP/DataLoader options <<<
    cfg["training"]["use_amp"]               = True
    cfg["training"]["amp_dtype"]             = "fp16"   # si ta carte gère bfloat16, tu peux mettre "bf16"
    cfg["training"]["grad_accumulation_steps"]= 1
    cfg["training"]["num_workers"]           = 2
    cfg["training"]["pin_memory"]            = True
    cfg["training"]["persistent_workers"]    = True

    # Sorties Dataiku (→ run_training écrira dans ces datasets)
    cfg["outputs"]["features_dataset"]        = None  # pas d’écriture features ici
    cfg["outputs"]["predictions_dataset"]     = OUTPUT_PREDICTIONS_DATASET
    cfg["outputs"]["metrics_dataset"]         = OUTPUT_METRICS_DATASET
    cfg["outputs"]["model_artifacts_dataset"] = OUTPUT_ARTIFACTS_DATASET
    cfg["outputs"]["local_dir"]               = "output"

    # Runtime
    cfg["runtime"]["verbose"]  = True
    cfg["runtime"]["progress"] = True
    cfg["runtime"]["seed"]     = RANDOM_SEED
    cfg["runtime"]["device_override"] = "cuda"     # force le GPU sur la recipe
    cfg["runtime"]["debug_limit_clients"] = DEBUG_LIMIT_CLIENTS

    log.info(f"Archi: {cfg['model']['n_layers']}L-{cfg['model']['n_heads']}H-{cfg['model']['d_model']}D")
    log.info(f"Seq:   {cfg['sequence']['months_lookback']} mois | horizon={cfg['sequence']['target_horizon']}")

    # 3) Train
    t0 = time.time()
    results = run_training(cfg)
    t_train = time.time() - t0
    log.info(f"Training done in {t_train:.1f}s")

    # 4) Log console rapide
    m  = results.get("metrics", {})
    mi = results.get("model_info", {})
    di = results.get("data_info", {})
    log.info(f"acc={m.get('accuracy',0):.4f} | prec={m.get('precision',0):.4f} | rec={m.get('recall',0):.4f} | f1={m.get('f1',0):.4f}")
    log.info(f"Model: {mi.get('architecture')} | params≈ {mi.get('total_params'):,}")
    log.info(f"Data : clients={di.get('n_clients')} | seq_len={di.get('seq_len')} | classes={di.get('n_classes')}")

    log.info("=== RECIPE 2 DONE ===")


if __name__ == "__main__":
    main()

