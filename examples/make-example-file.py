from ase.build import bulk
from ase.io import write
import numpy as np


def main():

    rng = np.random.default_rng(seed=0)

    tungsten_supercell = bulk("W", cubic=True).repeat((4, 4, 4))
    
    trajectory = []
    for _ in range(10):
        solution = tungsten_supercell.copy()
        solution.symbols = rng.choice(["W", "Re"], size=len(solution), p=[0.8, 0.2])
        trajectory.append(solution)
    write("examples/solution.xyz", trajectory, format="extxyz")


if __name__ == "__main__":

    main()
