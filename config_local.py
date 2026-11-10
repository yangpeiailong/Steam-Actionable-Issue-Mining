"""Local paths for Steam Actionable Issue Mining.

Set these environment variables before running scripts:
  STEAM_DATA_ROOT   Root containing Paper1_Actionable_Issue_Mining/
  STEAM_MODELS_DIR Root containing base_models/ and stage2_runs/

No private paths or credentials are stored in this file.
"""
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent

# Expected data layout:
# <STEAM_DATA_ROOT>/Paper1_Actionable_Issue_Mining/data/...
# <STEAM_DATA_ROOT>/Paper1_Actionable_Issue_Mining/results/...
STEAM_DATA_ROOT = Path(
    os.environ.get("STEAM_DATA_ROOT", REPO_ROOT / "data_local")
).expanduser().resolve()

PAPER_ROOT = STEAM_DATA_ROOT
SHARED_DATA_ROOT = PAPER_ROOT / "shared_data"
SHARED_RAW_ROOT = SHARED_DATA_ROOT / "raw"
SHARED_CLEANED_ROOT = SHARED_DATA_ROOT / "cleaned"
SHARED_TAXONOMY_ROOT = SHARED_DATA_ROOT / "common_taxonomy"
SHARED_METADATA_ROOT = SHARED_DATA_ROOT / "common_metadata"

PAPER1_ROOT = PAPER_ROOT / "Paper1_Actionable_Issue_Mining"
PAPER1_DATA_ROOT = PAPER1_ROOT / "data"
PAPER1_ANALYSIS_ROOT = PAPER1_DATA_ROOT / "analysis"
PAPER1_RESULTS_ROOT = PAPER1_ROOT / "results"

# Expected model layout:
# <STEAM_MODELS_DIR>/base_models/deberta/
# <STEAM_MODELS_DIR>/stage2_runs/<run_name>/best_model/
MODELS_ROOT = Path(
    os.environ.get("STEAM_MODELS_DIR", REPO_ROOT / "models_local")
).expanduser().resolve()
BASE_MODELS_ROOT = MODELS_ROOT / "base_models"
STAGE2_RUNS_ROOT = MODELS_ROOT / "stage2_runs"
MODEL_RUNS_ROOT = STAGE2_RUNS_ROOT
DEBERTA_BASE_MODEL_DIR = Path(
    os.environ.get(
        "STEAM_DEBERTA_BASE_MODEL_DIR",
        BASE_MODELS_ROOT / "deberta"
    )
).expanduser().resolve()

def _model_dir(env_name: str, default_name: str) -> Path:
    return Path(
        os.environ.get(env_name, STAGE2_RUNS_ROOT / default_name / "best_model")
    ).expanduser().resolve()

# Final analysis models (frozen checkpoints; weights are not distributed here).
STAGE2_ROBERTA_T2_DIR = _model_dir(
    "STEAM_ROBERTA_T2_DIR", "roberta_targeted_T2_arch_v1"
)
STAGE2_DEBERTA_T3_DIR = _model_dir(
    "STEAM_DEBERTA_T3_DIR", "targeted_T3_v1"
)

# Checkpoints used in the locked test-set model comparison scripts.
DEBERTA_GOLD_ONLY_DIR = _model_dir(
    "STEAM_DEBERTA_GOLD_ONLY_DIR", "deberta_stage2_gold_only_repro_v1"
)
DEBERTA_HARD_EXPANDED_1289_DIR = _model_dir(
    "STEAM_DEBERTA_HARD_EXPANDED_1289_DIR",
    "deberta_stage2_distill_hard_expanded_1289_v1"
)
ROBERTA_GOLD_ONLY_DIR = _model_dir(
    "STEAM_ROBERTA_GOLD_ONLY_DIR", "roberta_gold_only_arch_v1"
)
MODERNBERT_GOLD_ONLY_DIR = _model_dir(
    "STEAM_MODERNBERT_GOLD_ONLY_DIR", "modernbert_gold_only_arch_v1"
)
MODERNBERT_TARGETED_T1_DIR = _model_dir(
    "STEAM_MODERNBERT_TARGETED_T1_DIR", "modernbert_targeted_T1_arch_v1"
)
