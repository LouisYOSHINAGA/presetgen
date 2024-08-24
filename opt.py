from deap import creator, base, tools, algorithms
import numpy as np
from typing import TypeAlias
from hparam import HyperParams
from util import Graycode, i2g, g2i
from obj import objective, plot_pareto_front


Parameters: TypeAlias = list[float]
Individual: TypeAlias = Graycode
Population: TypeAlias = list[Individual]


def encode(hps: HyperParams) -> Individual:
    individual: Individual = []
    for _ in range(hps.n_params):
        param_int_bit: int = np.random.default_rng().integers(0, hps.n_param_step_bit)  # sample from [0, hps.n_param_step_bit)
        individual.extend(i2g(param_int_bit, hps.n_param_bit))
    return individual

def decode(individual: Individual, hps: HyperParams) -> Parameters:
    parameters: Parameters = []
    sbit: int = 0
    for _ in range(hps.n_params):
        bits: int = hps.n_param_bit
        param_int: float = round(hps.n_param_step * g2i(individual[sbit:sbit+bits]) / hps.n_param_step_bit)
        parameters.append(hps.param_ranges[0] + hps.param_ranges[2] * param_int)
        sbit += bits
    return parameters

def initialize(hps: HyperParams) -> tuple[base.Toolbox, tools.Statistics, tools.Logbook, tools.ParetoFront]:
    creator.create("FitnessMulti", base.Fitness, weights=hps.directions)
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("parameters", encode, hps=hps)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.parameters)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=hps.n_population)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1/len(encode(hps)))
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", objective, hps=hps)

    stats = tools.Statistics(lambda individual: individual.fitness.values)
    stats.register("minimize", min)
    stats.register("maximize", max)
    logbook = tools.Logbook()
    logbook.header = ("generation", "minimize", "maximize")
    pareto_front = tools.ParetoFront()

    return toolbox, stats, logbook, pareto_front

def optimize(hps: HyperParams) -> None:
    toolbox, stats, logbook, pareto_front = initialize(hps)

    population: Population = toolbox.population()  # P(t=0)
    evaluate(toolbox, population, None, hps)

    for generation in range(hps.n_generation):
        offspring: Population = algorithms.varAnd(population, toolbox, cxpb=hps.p_mate, mutpb=hps.p_mutate)  # Q(t)

        evaluate(toolbox, offspring, pareto_front, hps)
        logbook.record(generation=generation, **stats.compile(offspring))
        print(f"{logbook.stream}")

        population[:] = toolbox.select(population+offspring, k=hps.n_population)  # P(t+1)

    finalize(pareto_front, hps)

def evaluate(toolbox: base.Toolbox, individuals: Population, pareto_front: tools.ParetoFront|None, hps: HyperParams) -> None:
    for individual in individuals:
        if not individual.fitness.valid:
            individual.fitness.values = toolbox.evaluate(decode(individual, hps))
    if pareto_front is not None:
        pareto_front.update(individuals)

def finalize(pareto_front: tools.ParetoFront, hps: HyperParams) -> None:
    print(f"Pareto Front:")
    for i, individual in enumerate(pareto_front):
        print(f"{i:03d}: ", end="")

        print(f"parameter = [", end="")
        params: Parameters = decode(individual, hps)
        for j, param in enumerate(params):
            print(f"{param:05f}", end="")
            if j < len(params) - 1:
                print(f", ", end="")
            else:
                print(f"], ", end="")

        print(f"objective = [", end="")
        for j, obj in enumerate(individual.fitness.values):
            print(f"{obj:05f}", end="")
            if j < len(individual.fitness.values) - 1:
                print(f", ", end="")
            else:
                print(f"]")

    plot_pareto_front(pareto_front)