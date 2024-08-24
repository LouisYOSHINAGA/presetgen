from deap import tools
import numpy as np
from typing import TypeAlias, Any
import matplotlib.pyplot as plt
from hparam import HyperParams


ObjectiveValue: TypeAlias = float
ObjectiveValues: TypeAlias = tuple[ObjectiveValue, ObjectiveValue]


def objective(params: list[Any], hps: HyperParams) -> ObjectiveValues:
    return simple_multi_objective(params)

def simple_multi_objective(params: list[float]) -> ObjectiveValues:
    return (float(np.sum(params)), float(np.prod(params)))


def plot_pareto_front(pareto_front: tools.ParetoFront) -> None:
    plt.figure(figsize=(6, 6))
    plt.xlabel("Objective 0")
    plt.ylabel("Objective 1")
    for individual in pareto_front:
        plt.scatter(*individual.fitness.values, color=plt.get_cmap("tab10")(0))
    plt.show()