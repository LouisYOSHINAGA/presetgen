import datetime, zoneinfo, os
import logging
from typing import Any
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from hparam import HyperParams
from obj import ObjectiveValues


class OptRecorder:
    def __init__(self, keys: list[str]) -> None:
        self.recorder: dict[str, list[Any]] = {key: [] for key in keys}

    def __call__(self, **contents: dict[str, Any]) -> None:
        for key, value in contents.items():
            self.recorder[key].append(value)

    def get(self, key: str) -> list[Any]:
        return self.recorder[key]


def get_time(format: str ="%Y_%m%d_%H%M", is_reset: bool =False) -> str:
    if is_reset or not hasattr(get_time, "time"):
        d = datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Tokyo"))
        get_time.time = str(d.strftime(format))
    return get_time.time

def get_file_path(hps: HyperParams, header="", ext="") -> str:
    log_dir: str = f"{hps.log_dir}/log_{get_time()}"
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    return f"{log_dir}/{header}{get_time()}{ext}"

def setup_logger(hps: HyperParams, name: str, terminator: str ="") -> logging.Logger:
    handler = logging.FileHandler(get_file_path(hps, header="log_", ext=".log"))
    handler.setLevel(logging.INFO)
    handler.terminator = terminator

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def save_history(recorder: OptRecorder, key: str, xlabel: str, ylabel: str|list[str], hps: HyperParams) -> None:
    contents: list[Any] = recorder.get(key)
    n_elements: int = len(contents[0])
    plt.figure(figsize=(15, 3*n_elements))
    for i in range(n_elements):
        plt.subplot(n_elements, 1, i+1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel if isinstance(ylabel, str) else ylabel[i])
        plt.plot([content[i] for content in contents])
    plt.tight_layout()
    save_path: str = get_file_path(hps, header=f"history_{key}_", ext=".png")
    plt.savefig(save_path, dpi=320)
    hps.logger.info(f"History of {key} is saved in '{save_path}'.")

def save_scatter_matrix(objss: list[ObjectiveValues], hps: HyperParams) -> None:
    data = pd.DataFrame(objss, columns=hps.objectives)
    p = sns.pairplot(data)
    save_path: str = get_file_path(hps, header="scatter_matrix_", ext=".png")
    p.savefig(save_path, dpi=320)
    hps.logger.info(f"Scatter matrix of objectives is saved in '{save_path}'.")