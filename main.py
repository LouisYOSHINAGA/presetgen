import fire
from typing import Any
from hparam import setup_hyperparams
from opt import optimize


def run(**kwargs: Any):
    hps = setup_hyperparams(**kwargs)
    optimize(hps)


if __name__ == "__main__":
    fire.Fire(run)