from typing import TypeAlias, Literal


Bit: TypeAlias = Literal[0, 1]
Graycode: TypeAlias = list[Bit]


def i2g(n: int, length: int) -> Graycode:
    return [(n ^ (n >> 1)) >> i & 1 for i in reversed(range(length))]

def g2i(g: Graycode) -> int:
    n = m = g[0]
    for i in range(1, len(g)):
        m ^= g[i]
        n = (n << 1) + m
    return n


if __name__ == "__main__":
    ints = [4095, 4096, 2048]
    dig = 13

    for i in ints:
        print(f"{i} -> {i2g(i, dig)} -> {g2i(i2g(i, dig))}")