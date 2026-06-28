from ovito.io import import_file
from tce_modifier import TCEModifier
from ase.build import bulk
import numpy as np


def main():

    lattice_parameter = 3.16
    bcc_cutoffs = np.array([0.5 * np.sqrt(3.0), 1.0, np.sqrt(2.0)])

    pipeline = import_file("examples/solution.xyz")
    pipeline.modifiers.append(
        TCEModifier(
            neighbor_cutoffs=lattice_parameter * bcc_cutoffs,
            many_body_features=[(0, 0, 1), (0, 0, 2)],
            species=["W", "Re"],
        )
    )
    data = pipeline.compute()
    print(data.attributes)


if __name__ == "__main__":

    main()