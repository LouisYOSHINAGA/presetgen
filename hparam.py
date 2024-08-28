import numpy as np
from typing import Any


class HyperParams(dict):
    def __getattr__(self, key: str) -> Any:
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state


default_hps = HyperParams(
    directions=(-1, -1, +1),
    objectives=['sum', 'prod', 'max'],

    n_params=3,
    param_ranges=(-1, 1, 0.01),

    p_mate=0.6,
    p_mutate=1.0,

    n_population=8,
    n_generation=50,

    log_dir="../log",
)


def setup_hyperparams(**kwargs: dict[str, Any]) -> HyperParams:
    hps: HyperParams = default_hps
    for k, v in kwargs.items():
        assert k in hps, f"Hyper Parameter '{k}' does not exist."
        hps[k] = v

    hps.n_param_step = len(np.arange(*hps.param_ranges))
    hps.n_param_bit = int(np.ceil(np.log2(hps.n_param_step)))
    hps.n_param_step_bit = 2 ** hps.n_param_bit
    return hps
