import os
import sys
from pathlib import Path
import toml

MAIN_PATH = Path(__file__).resolve().parents[1]
HOME_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_PATH = MAIN_PATH / "analysis"
DATA_PATH = MAIN_PATH / "data"
LOGS_PATH = MAIN_PATH / "logs"
BASE_CKPT_DIR = MAIN_PATH / "ckpts"
PYCHARMM_DIR = None

# check for paths.toml in main directory
if not HOME_DIR.joinpath("paths.toml").exists():
    raise FileNotFoundError(
        f"paths.toml not found in {HOME_DIR}. Please create the file with the required paths."
    )
    # read the paths.toml file
    paths = toml.load(HOME_DIR / "paths.toml")

    if "data" in paths:
        DATA_PATH = Path(paths["data"])
    if "logs" in paths:
        LOGS_PATH = Path(paths["logs"])
    if "analysis" in paths:
        ANALYSIS_PATH = Path(paths["analysis"])
    if "main" in paths:
        MAIN_PATH = Path(paths["main"])
    if "home" in paths:
        HOME_DIR = Path(paths["home"])
    if "pycharm" in paths:
        PYCHARMM_DIR = Path(paths["pycharm"])
    if "checkpoints" in paths:
        BASE_CKPT_DIR = Path(paths["checkpoints"])