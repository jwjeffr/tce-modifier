from ovito.io import import_file
from tce_modifier import TCEModifier
from tce.calculator import TCECalculator
from ase.build import bulk
import numpy as np


def main():

    lattice_parameter = 3.16
    modifier = TCEModifier(
        neighbor_cutoffs=[
            0.5 * np.sqrt(3.0) * lattice_parameter, 
            1.0 * lattice_parameter, 
            np.sqrt(2.0) * lattice_parameter
        ],
        many_body_features=[
            [0, 0, 1], [0, 0, 2],
            [0, 0, 0, 0, 1, 1]
        ],
        species=["W", "Re"]
    )

    pipeline = import_file("examples/solution.xyz")
    pipeline.modifiers.append(modifier)

    for data in pipeline.frames:
        print(data.attributes)


if __name__ == "__main__":

    main()