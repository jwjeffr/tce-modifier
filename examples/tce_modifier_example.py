from ovito.io import import_file
from tce_modifier import TCEModifier
from tce.calculator import TCECalculator
from ase.build import bulk
import numpy as np


def main():

    lattice_parameter = 3.16
    bcc_cutoffs = np.array([0.5 * np.sqrt(3.0), 1.0, np.sqrt(2.0)])
    calc = TCECalculator(
        neighbor_cutoffs=lattice_parameter * bcc_cutoffs,
        many_body_features=[
            (0, 0, 1), (0, 0, 2),
            (0, 0, 0, 0, 1, 1)
        ],
        species=["W", "Re"]
    )

    pipeline = import_file("examples/solution.xyz")
    pipeline.modifiers.append(TCEModifier(calc=calc))

    for data in pipeline.frames:
        print(data.attributes)


if __name__ == "__main__":

    main()