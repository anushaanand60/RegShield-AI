from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

RAW_DATA_PATH   = ROOT_DIR / "Data" / "bank_of_india_dataset.csv"
ARTIFACT_DIR    = ROOT_DIR / "Outputs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "F3924"

CATEGORICAL_COLS = [
    "F2230",  # month-year bucket (Sep25/Oct25/Nov25/Dec25) - DROPPED TO PREVENT LEAK
    "F3886",  # account type (Savings/Current/...)
    "F3888",  # date string (DD-MM-YYYY)
    "F3889",  # tenure bucket
    "F3890",  # geography code (R/U/SU/M)
    "F3891",  # occupation
    "F3892",  # gender
    "F3893",  # segment (RETAIL)
]

DOMINANT_FLAG_COL = "F3912"

HINT_FEATURES = [
    "F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
    "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3889", "F3891", "F3894",
]

TENURE_ORDER = ["L7D", "L31D", "L90D", "L180D", "L365D", "G365D"]
TENURE_MAP   = {v: i + 1 for i, v in enumerate(TENURE_ORDER)}
MONTH_MAP    = {"Sep25": 1, "Oct25": 2, "Nov25": 3, "Dec25": 4}

RANDOM_STATE = 42
N_SPLITS = 5
HIGH_NULL_THRESHOLD = 0.30