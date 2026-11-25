"""
config.py
---------
Central configuration for the HR Attrition Analytics platform.
Keeps paths, constants, and tunable settings in one place so no
"magic strings" are duplicated across modules.
"""

import os
import logging

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
EDA_DIR = os.path.join(BASE_DIR, "eda_plots")
EVAL_DIR = os.path.join(BASE_DIR, "eval_plots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, EDA_DIR, EVAL_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, "employee_attrition.csv")

# ------------------------------------------------------------------
# Dataset schema
# ------------------------------------------------------------------
TARGET_COL = "Attrition"
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Risk segmentation thresholds (predicted probability of leaving)
RISK_THRESHOLDS = {"Low Risk": 0.30, "Medium Risk": 0.60}

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger that logs to console + a shared file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(LOGS_DIR, "pipeline.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
