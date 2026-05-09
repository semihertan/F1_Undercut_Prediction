from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = OUTPUTS_DIR / "models"
CACHE_DIR = ROOT_DIR / "f1_cache"

DATA_FILE = RAW_DIR / "f1_22_25_undercut_data.csv"
SAMPLE_FILE = SAMPLE_DIR / "sample_race_snapshot.csv"
MODEL_FILE = MODELS_DIR / "f1_undercut_model.pkl"

FEATURES = ["Gap", "Driver_TyreLife", "Ahead_TyreLife", "Tyre_Advantage"]


def ensure_output_dirs() -> None:
    for directory in (RAW_DIR, SAMPLE_DIR, FIGURES_DIR, RESULTS_DIR, REPORTS_DIR, MODELS_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
