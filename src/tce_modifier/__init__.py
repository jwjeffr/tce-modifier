#### Python Modifier Name ####
# Description of your Python-based modifier.

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import factorial
from itertools import permutations, combinations
from colorsys import hsv_to_rgb

from ovito.data import DataCollection, DataTable
from ovito.pipeline import ModifierInterface
from ovito.io.ase import ovito_to_ase
from ovito.traits import action_handler
from traits.api import List, Float, Int, String, Bool
from tce.calculator import TCECalculator
from numpy.typing import NDArray
import numpy as np
from multiset import FrozenMultiset


class TCEModifier(ModifierInterface):

    neighbor_cutoffs = List(
        Float, 
        value=[3.0], 
        label="Neighbor Cutoffs", 
        minlen=1,
        ovito_group="Calculation"
    )
    many_body_features = List(
        List(Int), 
        value=[], 
        label="Many Body Features",
        ovito_group="Calculation"
    )
    neighbor_tolerance = Float(
        value=0.01, 
        label="Neighbor tolerance",
        ovito_group="Calculation"
    )
    visualize_cluster = Int(
        value=-1, 
        label="Visualize cluster row (-1 = off)",
        ovito_group="Visualization"
    )
    cluster_bond_width = Float(
        value=0.15, 
        label="Cluster bond width",
        ovito_group="Visualization"
    )
    select_visualized_cluster = Bool(
        value=False, 
        label="Select visualized cluster",
        ovito_group="Visualization"
    )
    unwrap_cluster_bonds = Bool(
        value=True,
        label="Unwrap cluster bonds through PBC",
        ovito_group="Visualization"
    )
    
    @staticmethod
    def _cluster_instance_color(cluster_id: int):
        """
        Deterministic color for one physical cluster instance.

        Uses a golden-ratio hue step so nearby cluster IDs get visually
        separated colors.
        """
        hue = (cluster_id * 0.618033988749895) % 1.0
        return np.asarray(hsv_to_rgb(hue, 0.75, 1.0), dtype=float)
        
    def _tensors_for_cluster_key(self, calc, topological_tensors, topology_key, body_order):
        """
        Return the topological tensor block(s) corresponding to one displayed
        aggregate cluster row.
        """
        if body_order == 2:
            neighbor_order = int(next(iter(topology_key)))
            return [topological_tensors[2][neighbor_order]]

        tensors = []

        for feature_index, feature in enumerate(calc.feature_groups[body_order]):
            if FrozenMultiset(feature) == topology_key:
                tensors.append(topological_tensors[body_order][feature_index])

        return tensors
        
    def _pbcvec_between_particles(self, data, a, b):
        """
        Return OVITO's Periodic Image vector for a bond a -> b.

        This wraps Particles.delta_vector(), but includes a small sign sanity
        check against OVITO's documented bond-vector convention:

            v_ab = x_b - x_a + H @ pbcvec
        """
        delta, pbcvec = data.particles.delta_vector(
            int(a),
            int(b),
            data.cell,
            return_pbcvec=True,
        )

        pbcvec = np.rint(pbcvec).astype(np.int32)

        # Defensive sign check. OVITO's bond convention is:
        #   x_b - x_a + H @ n
        # where n is the Periodic Image vector.
        positions = np.asarray(data.particles.positions)
        H = np.asarray(data.cell[:3, :3])

        direct_plus = positions[int(b)] - positions[int(a)] + H @ pbcvec
        direct_minus = positions[int(b)] - positions[int(a)] - H @ pbcvec

        if np.linalg.norm(direct_minus - delta) < np.linalg.norm(direct_plus - delta):
            pbcvec = -pbcvec

        return pbcvec
    
    def _cluster_site_images(self, data, cluster_sites):
        """
        Assign a consistent periodic image vector to every atom in one physical
        cluster instance.

        The first atom is the anchor with image vector (0, 0, 0).
        Every other atom is placed in the nearest periodic image relative to
        that anchor.

        Returns:
            dict[int, np.ndarray[int32]]
        """
        cluster_sites = tuple(int(i) for i in cluster_sites)

        root = cluster_sites[0]
        site_images = {
            root: np.zeros(3, dtype=np.int32),
        }

        for site in cluster_sites[1:]:
            site_images[site] = self._pbcvec_between_particles(
                data=data,
                a=root,
                b=site,
            )

        return site_images
        
    def _create_cluster_instance_bonds(self, data, atoms, calc, cluster_keys):
        """
        Visualize one selected cluster type.

        Each physical cluster instance of that type gets its own color.
        Each instance is drawn as a complete graph over the participating atoms.
        """
        if self.visualize_cluster < 0:
            return

        if self.visualize_cluster >= len(cluster_keys):
            print(
                f"TCE warning: visualize_cluster={self.visualize_cluster} "
                f"is out of range. Valid rows are 0 to {len(cluster_keys) - 1}."
            )
            return

        topology_key, species_key = cluster_keys[self.visualize_cluster]
        body_order = len(species_key)

        topological_tensors = calc.get_topological_tensors(atoms)
        symbols = np.asarray(atoms.get_chemical_symbols())

        tensors = self._tensors_for_cluster_key(
            calc=calc,
            topological_tensors=topological_tensors,
            topology_key=topology_key,
            body_order=body_order,
        )

        # Maps canonical physical cluster -> accumulated weight.
        #
        # Example:
        #   ordered tensor entries:
        #       (4, 10, 12)
        #       (4, 12, 10)
        #       (10, 4, 12)
        #       ...
        #
        # all map to:
        #       (4, 10, 12)
        physical_clusters = defaultdict(float)

        for tensor in tensors:
            coords = np.asarray(tensor.coords)
            values = np.asarray(tensor.data, dtype=float)

            if coords.size == 0:
                continue

            for site_indices, value in zip(coords.T, values):
                site_indices = np.asarray(site_indices, dtype=int)

                # Skip degenerate tuples such as (i, i, j), if any.
                if len(set(site_indices)) != body_order:
                    continue

                # Check the species multiset, ignoring order.
                if FrozenMultiset(str(s) for s in symbols[site_indices]) != species_key:
                    continue

                # Canonical physical cluster identity.
                cluster_sites = tuple(sorted(int(i) for i in site_indices))

                # Divide out ordered permutations.
                physical_clusters[cluster_sites] += float(value) / factorial(body_order)

        if not physical_clusters:
            return

        bond_topology = []
        bond_colors = []
        bond_cluster_ids = []
        bond_cluster_weights = []
        bond_pbc_vectors = []

        for cluster_id, (cluster_sites, cluster_weight) in enumerate(
            sorted(physical_clusters.items())
        ):
            cluster_sites = tuple(int(i) for i in cluster_sites)
            color = self._cluster_instance_color(cluster_id)

            if self.unwrap_cluster_bonds and data.cell is not None:
                site_images = self._cluster_site_images(data, cluster_sites)
            else:
                site_images = {
                    site: np.zeros(3, dtype=np.int32)
                    for site in cluster_sites
                }

            for a, b in combinations(cluster_sites, 2):
                a = int(a)
                b = int(b)

                bond_topology.append([a, b])
                bond_colors.append(color)
                bond_cluster_ids.append(cluster_id)
                bond_cluster_weights.append(cluster_weight)

                # Important:
                # Do NOT compute delta_vector(a, b) independently here.
                # Use the cluster-consistent image assignment.
                pbcvec = site_images[b] - site_images[a]
                bond_pbc_vectors.append(pbcvec)

        bonds = data.particles_.create_bonds(count=len(bond_topology))

        bonds.create_property(
            "Topology",
            data=np.asarray(bond_topology, dtype=np.int64),
        )

        bonds.create_property(
            "Color",
            data=np.asarray(bond_colors, dtype=float),
        )

        bonds.create_property(
            "TCE Cluster ID",
            data=np.asarray(bond_cluster_ids, dtype=np.int32),
        )

        bonds.create_property(
            "TCE Cluster Weight",
            data=np.asarray(bond_cluster_weights, dtype=float),
        )

        bonds.create_property(
            "Periodic Image",
            data=np.asarray(bond_pbc_vectors, dtype=np.int32),
        )

        bonds.vis.enabled = True
        bonds.vis.width = self.cluster_bond_width

        data.attributes["TCE.visualized_cluster_row"] = int(self.visualize_cluster)
        data.attributes["TCE.num_visualized_clusters"] = len(physical_clusters)
        data.attributes["TCE.num_visualization_bonds"] = len(bond_topology)


    def _cluster_participation(self, calc, atoms, topology_key, species_key):
        """
        Return a per-atom participation count for one aggregated cluster type.

        topology_key and species_key are the FrozenMultiset objects used as keys
        in unique_clusters.

        The returned array has length len(atoms). Values are nonzero for atoms
        participating in the selected cluster type.
        """
        topological_tensors = calc.get_topological_tensors(atoms)

        body_order = len(species_key)
        symbols = np.asarray(atoms.get_chemical_symbols())

        participation = np.zeros(len(atoms), dtype=float)

        # Species are aggregated as a multiset in your table, so we should accept
        # any permutation of the species tuple.
        species_tuple = tuple(str(s) for s in species_key)
        allowed_species_orders = set(permutations(species_tuple))

        tensors_to_visualize = []

        if body_order == 2:
            # For pair features, topology_key is something like FrozenMultiset((0,))
            # where 0 is the neighbor cutoff index.
            neighbor_order = int(list(topology_key)[0])
            tensors_to_visualize.append(topological_tensors[2][neighbor_order])

        else:
            # For many-body features, find the stored tensor(s) whose topology
            # label matches this aggregated topology multiset.
            for feature_index, feature in enumerate(calc.feature_groups[body_order]):
                if FrozenMultiset(feature) == topology_key:
                    tensors_to_visualize.append(
                        topological_tensors[body_order][feature_index]
                    )

        for tensor in tensors_to_visualize:
            coords = np.asarray(tensor.coords)
            values = np.asarray(tensor.data, dtype=float)

            if coords.size == 0:
                continue

            # coords has shape:
            #   body_order x nnz
            #
            # Each column is one ordered cluster occurrence:
            #   pair:      i, j
            #   triplet:   i, j, k
            #   quadruplet i, j, k, l
            #
            # The tensors are symmetric/ordered, so each physical unordered
            # cluster appears multiple times. Dividing by factorial(body_order)
            # makes the per-atom value closer to "number of physical clusters"
            # rather than "number of tensor permutations".
            for site_indices, value in zip(coords.T, values):
                site_indices = np.asarray(site_indices, dtype=int)

                if tuple(symbols[site_indices]) not in allowed_species_orders:
                    continue

                for atom_index in set(site_indices):
                    participation[int(atom_index)] += (
                        float(value) / factorial(body_order)
                    )

        return participation

    def modify(self, data: DataCollection, frame: int, input_slots, data_cache, **kwargs):
        
        atoms = ovito_to_ase(data)

        if "calc" not in data_cache.attributes:
            trajectory_species = set()

            for frame_index in range(input_slots["upstream"].num_frames):
                frame_data = input_slots["upstream"].compute(frame_index)
                frame_atoms = ovito_to_ase(frame_data)
                trajectory_species |= set(frame_atoms.get_chemical_symbols())

            trajectory_species = sorted(trajectory_species)

            print("Detected species:", trajectory_species)

            data_cache.attributes["calc"] = TCECalculator(
                neighbor_cutoffs=np.array(self.neighbor_cutoffs),
                many_body_features=self.many_body_features,
                species=trajectory_species,
                neighbor_tolerance=self.neighbor_tolerance
            )

        calc = data_cache.attributes["calc"]

        cluster_vector = calc.get_feature_vector(atoms)
        names = calc.get_feature_label_order()

        unique_clusters = defaultdict(float)

        for (topology, species), feature in zip(names, cluster_vector):
            topology = FrozenMultiset(topology)
            species = FrozenMultiset(species)
            unique_clusters[(topology, species)] += feature.item() / factorial(len(species))
            
        cluster_keys = list(unique_clusters.keys())

        self._create_cluster_instance_bonds(
            data=data,
            atoms=atoms,
            calc=data_cache.attributes["calc"],
            cluster_keys=cluster_keys,
        )

        # Keep a deterministic list of the displayed cluster rows.
        cluster_keys = list(unique_clusters.keys())

        # New visualization output:
        # Creates a scalar per-particle property named "TCE Participation".
        if 0 <= self.visualize_cluster < len(cluster_keys):
            topology_key, species_key = cluster_keys[self.visualize_cluster]

            participation = self._cluster_participation(
                calc=calc,
                atoms=atoms,
                topology_key=topology_key,
                species_key=species_key,
            )

            data.particles_.create_property(
                "TCE Participation",
                data=participation,
            )

            if self.select_visualized_cluster:
                data.particles_.create_property(
                    "Selection",
                    data=(participation > 0).astype(np.int32),
                )

            data.attributes["TCE.visualized_cluster_row"] = self.visualize_cluster
            data.attributes["TCE.visualized_cluster_count"] = float(
                unique_clusters[(topology_key, species_key)]
            )

        elif self.visualize_cluster != -1:
            print(
                f"TCE warning: visualize_cluster={self.visualize_cluster} "
                f"is out of range. Valid rows are 0 to {len(cluster_keys) - 1}."
            )

        table = data.tables.create(
            identifier="cluster counts",
            plot_mode=DataTable.PlotMode.BarChart,
            title="Cluster Counts"
        )

        table.x = table.create_property(
            "Cluster Type",
            data=np.arange(len(unique_clusters))
        )

        for i, ((top, species), feature) in enumerate(unique_clusters.items()):
            
            species = dict(species)
            cluster_chemical_formula = ''.join(f'{symbol}{count}' for symbol, count in species.items())
            
            top = dict(top)
            topology_formula = '\n'.join(f'{bond + 1}nn x{mult}' for bond, mult in top.items())
            
            feature_label_str = f"{cluster_chemical_formula}\n{topology_formula}"
            table.x.add_type_id(i, table, name=f"[{i}]\n{feature_label_str}")
            data.attributes[
                f"{feature_label_str.replace('\n', ' + ')} (ID = {i})"
            ] = feature

        table.y = table.create_property(
            "Count",
            data=np.array(list(unique_clusters.values()))
        )
