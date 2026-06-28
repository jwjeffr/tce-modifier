#### Python Modifier Name ####
# Description of your Python-based modifier.

from collections import Counter

from ovito.data import DataCollection, DataTable
from ovito.pipeline import ModifierInterface
from ovito.io.ase import ovito_to_ase
from tce.calculator import TCECalculator
from numpy.typing import NDArray
import numpy as np


class TCEModifier(ModifierInterface):

    neighbor_cutoffs: NDArray[np.floating]
    many_body_features: list[tuple[int, int, int]]
    species: list[str]

    def modify(self, data: DataCollection, frame: int, **kwargs):
        
        atoms = ovito_to_ase(data)
        calc = TCECalculator(
            neighbor_cutoffs=self.neighbor_cutoffs,
            many_body_features=self.many_body_features,
            species=self.species,
        )

        cluster_vector = calc.get_feature_vector(atoms)
        names = calc.get_feature_label_order()

        table = data.tables.create(
            identifier="cluster counts",
            plot_mode=DataTable.PlotMode.BarChart,
            title="Cluster Counts"
        )
        table.x = table.create_property("Cluster Type", data=np.arange(len(names)))
        for i, ((top, species), feature) in enumerate(zip(names, cluster_vector)):
            feature_label_str = f"{' & '.join(str(s) for s in top)}, {'-'.join(species)}"
            table.x.add_type_id(i, table, name=feature_label_str)
            data.attributes[feature_label_str] = feature.item()

        table.y = table.create_property("Count", data=cluster_vector)
