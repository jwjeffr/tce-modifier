from ase.build import bulk
from ase.io import write
import numpy as np


def main():

    rng = np.random.default_rng(seed=0)

    solution = bulk("W", cubic=True).repeat((4, 4, 4))
    solution.symbols = rng.choice(["W", "Re"], size=len(solution), p=[0.8, 0.2])
    write("examples/solution.xyz", solution, format="extxyz")


if __name__ == "__main__":

    main()
