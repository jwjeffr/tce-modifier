#### Python Modifier Name ####
# Description of your Python-based modifier.

from collections import Counter
from dataclasses import dataclass, field
from math import factorial

from ovito.data import DataCollection, DataTable
from ovito.pipeline import ModifierInterface
from ovito.io.ase import ovito_to_ase
from tce.calculator import TCECalculator
from numpy.typing import NDArray
import numpy as np
from multiset import FrozenMultiset


@dataclass
class TCEModifier(ModifierInterface):

    calc: TCECalculator

    def modify(self, data: DataCollection, frame: int, **kwargs):
        
        atoms = ovito_to_ase(data)

        cluster_vector = self.calc.get_feature_vector(atoms)
        names = self.calc.get_feature_label_order()

        unique_clusters = {}
        for (topology, species), feature in zip(names, cluster_vector):
            topology = FrozenMultiset(topology)
            species = FrozenMultiset(species)
            if (topology, species) in unique_clusters:
                continue
            unique_clusters[(topology, species)] = feature.item()

        table = data.tables.create(
            identifier="cluster counts",
            plot_mode=DataTable.PlotMode.BarChart,
            title="Cluster Counts"
        )
        table.x = table.create_property("Cluster Type", data=np.arange(len(unique_clusters)))
        for i, ((top, species), feature) in enumerate(unique_clusters.items()):
            feature_label_str = f"{' & '.join(str(s) for s in top)}, {'-'.join(species)}"
            table.x.add_type_id(i, table, name=feature_label_str)
            data.attributes[feature_label_str] = feature

        table.y = table.create_property("Count", data=np.array(list(unique_clusters.values())))
