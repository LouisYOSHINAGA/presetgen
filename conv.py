import numpy as np
from typing import TypeAlias, Literal
from hparam import HyperParams


Graycode: TypeAlias = list[Literal[0, 1]]
Individual: TypeAlias = Graycode
Population: TypeAlias = list[Individual]

Parameter: TypeAlias = int|float
Parameters: TypeAlias = list[Parameter]
ParametersSet: TypeAlias = list[Parameters]


def i2g(n: int, length: int) -> Graycode:
    return [(n ^ (n >> 1)) >> i & 1 for i in reversed(range(length))]

def g2i(g: Graycode) -> int:
    n = m = g[0]
    for i in range(1, len(g)):
        m ^= g[i]
        n = (n << 1) + m
    return n

def encode(hps: HyperParams) -> Individual:
    individual: Individual = []
    for _ in range(hps.n_params):
        param_int_bit: int = np.random.default_rng().integers(0, hps.n_param_step_bit)
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


if __name__ == "__main__":
    ints, dig = [4095, 4096, 2048], 13
    for i in ints:
        print(f"{i} -> {i2g(i, dig)} -> {g2i(i2g(i, dig))}")