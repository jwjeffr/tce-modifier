#### Python Modifier Name ####
# Description of your Python-based modifier.

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import factorial
from itertools import permutations

from ovito.data import DataCollection, DataTable
from ovito.pipeline import ModifierInterface
from ovito.io.ase import ovito_to_ase
from ovito.traits import action_handler
from traits.api import List, Float, Int, String
from tce.calculator import TCECalculator
from numpy.typing import NDArray
import numpy as np
from multiset import FrozenMultiset


class TCEModifier(ModifierInterface):

    neighbor_cutoffs = List(Float, value=[3.0], label="Neighbor Cutoffs", minlen=1)
    many_body_features = List(List(Int), value=[], label="Many Body Features")
    neighbor_tolerance = Float(value=0.01, label="Neighbor tolerance")

    def modify(self, data: DataCollection, frame: int, input_slots, data_cache, **kwargs):
        
        atoms = ovito_to_ase(data)

        if "calc" not in data_cache.attributes:
            
            trajectory_species = set()
            for frame_index in range(input_slots["upstream"].num_frames):
                frame_data = input_slots["upstream"].compute(frame_index)
                frame_atoms = ovito_to_ase(frame_data)
                trajectory_species |= set(frame_atoms.get_chemical_symbols())
                
            trajectory_species = sorted(trajectory_species)
            
            data_cache.attributes["calc"] = TCECalculator(
                neighbor_cutoffs=np.array(self.neighbor_cutoffs),
                many_body_features=self.many_body_features,
                species=sorted(trajectory_species),
                neighbor_tolerance=self.neighbor_tolerance
            )

        cluster_vector = data_cache.attributes["calc"].get_feature_vector(atoms)
        names = data_cache.attributes["calc"].get_feature_label_order()

        unique_clusters = defaultdict(float)
        for (topology, species), feature in zip(names, cluster_vector):
            topology = FrozenMultiset(topology)
            species = FrozenMultiset(species)
            unique_clusters[(topology, species)] += feature.item() / factorial(len(species))

        table = data.tables.create(
            identifier="cluster counts",
            plot_mode=DataTable.PlotMode.BarChart,
            title="Cluster Counts"
        )
        table.x = table.create_property("Cluster Type", data=np.arange(len(unique_clusters)))
        for i, ((top, species), feature) in enumerate(unique_clusters.items()):
            feature_label_str = f"{' & '.join(str(s) for s in top)}\n{'-'.join(species)}"
            table.x.add_type_id(i, table, name=feature_label_str)
            data.attributes[feature_label_str] = feature

        table.y = table.create_property("Count", data=np.array(list(unique_clusters.values())))
