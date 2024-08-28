import numpy as np
from typing import TypeAlias, Callable, Any
from hparam import HyperParams
from opt import Parameters


ObjectiveFunction: TypeAlias = Callable[[Parameters], float]
ObjectiveValue: TypeAlias = float
ObjectiveValues: TypeAlias = list[ObjectiveValue]
ObjectiveValuesSet: TypeAlias = list[ObjectiveValues]


def objective(params: list[Any], hps: HyperParams) -> ObjectiveValues:
    objective_fns: dict[str, ObjectiveFunction] = {
        'sum': np.sum,
        'prod': np.prod,
        'max': np.max,
    }
    return [objective_fns[obj](params) for obj in hps.objectives]