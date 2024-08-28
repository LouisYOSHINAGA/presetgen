from deap import creator, base, tools, algorithms
from hparam import HyperParams
from conv import Population, Parameters, ParametersSet, encode, decode
from obj import ObjectiveValues, ObjectiveValuesSet, objective
from util import OptRecorder, setup_logger, save_history, save_scatter_matrix


def initialize(hps: HyperParams) -> tuple[base.Toolbox, OptRecorder, tools.ParetoFront]:
    hps.logger = setup_logger(hps=hps, name=__name__)

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

    recorder = OptRecorder(keys=["params", "objs"])
    pareto_front = tools.ParetoFront()

    return toolbox, recorder, pareto_front

def optimize(hps: HyperParams) -> None:
    toolbox, recorder, pareto_front = initialize(hps)

    population: Population = toolbox.population()  # P(t=0)
    evaluate(toolbox, population, None, None, hps)

    for generation in range(hps.n_generation):
        offspring: Population = algorithms.varAnd(population, toolbox, cxpb=hps.p_mate, mutpb=hps.p_mutate)  # Q(t)
        evaluate(toolbox, offspring, recorder, pareto_front, hps)
        population[:] = toolbox.select(population+offspring, k=hps.n_population)  # P(t+1)

    finalize(recorder, pareto_front, hps)

def evaluate(toolbox: base.Toolbox, individuals: Population,
             recorder: OptRecorder|None, pareto_front: tools.ParetoFront|None,
             hps: HyperParams) -> None:
    for individual in individuals:
        params: Parameters = decode(individual, hps)
        if not individual.fitness.valid:
            individual.fitness.values = toolbox.evaluate(params)
        if recorder is not None:
            recorder(params=params, objs=individual.fitness.values)

    if pareto_front is not None:
        pareto_front.update(individuals)

def finalize(recorder: OptRecorder, pareto_front: tools.ParetoFront, hps: HyperParams) -> None:
    paramss: ParametersSet = []
    objss: ObjectiveValuesSet = []

    hps.logger.info(f"Pareto Front:\n")
    for i, individual in enumerate(pareto_front):
        hps.logger.info(f"{i:03d}-th:\n")

        params: Parameters = decode(individual, hps)
        paramss.append(params)

        hps.logger.info(f"  parameters = [")
        for j, param in enumerate(params):
            hps.logger.info(f"{param:05f}")
            if j < len(params) - 1:
                hps.logger.info(f", ")
            else:
                hps.logger.info(f"]\n")

        objs: ObjectiveValues = individual.fitness.values
        objss.append(objs)

        hps.logger.info(f"  objectives = [")
        for j, obj in enumerate(objs):
            hps.logger.info(f"{obj:05f}")
            if j < len(individual.fitness.values) - 1:
                hps.logger.info(f", ")
            else:
                hps.logger.info(f"]\n\n")

    save_history(recorder, key="params", xlabel="n_individual", ylabel="parameter value", hps=hps)
    save_history(recorder, key="objs", xlabel="n_individual", ylabel=hps.objectives, hps=hps)
    save_scatter_matrix(objss, hps)