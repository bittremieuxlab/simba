"""Molecule-set training batches: pick a small number of anchor molecules,
expand each via literal K-nearest-neighbor-by-MCES to reach n total unique
molecules, take 2 spectra per molecule (different indices when available),
and score the full (n, n) cosine-similarity cross matrix against the true
(n, n) MCES-derived target -- see
SimilarityModelMultitask.molecule_set_step. Training only; validation stays
on the existing per-pair CustomDatasetMultitasking pipeline, unchanged.
"""

import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from simba.core.data.datasets.multitask_dataset import CustomDatasetMultitasking


def build_dense_mces_matrix(pair_distances, extra_distances, n_molecules):
    """Scatter the exhaustive (idx0, idx1, mces) pair table already loaded
    in memory (molecule_pairs_train.pair_distances[:, :2] /
    .extra_distances -- see simba/workflows/training.py:prepare_data) into
    a dense symmetric (n_molecules, n_molecules) float32 matrix (0 on the
    diagonal), for O(1) per-molecule nearest-neighbor lookups. No disk
    re-read needed. ~2.3GB at n_molecules=24010; built once, kept resident
    for the whole run."""
    dense = np.zeros((n_molecules, n_molecules), dtype=np.float32)
    idx0 = pair_distances[:, 0].astype(np.int64)
    idx1 = pair_distances[:, 1].astype(np.int64)
    mces = np.asarray(extra_distances, dtype=np.float32).reshape(-1)
    dense[idx0, idx1] = mces
    dense[idx1, idx0] = mces
    return dense


class MoleculeSetDataset(IterableDataset):
    """Yields one fully-assembled molecule-set training batch per
    iteration -- use with DataLoader(dataset, batch_size=None, ...) since
    each yield is already a complete batch, not one item to be collated.

    Per batch: `n_anchors` anchor molecules drawn uniformly at random,
    each expanded to its `k_nearest` nearest unselected neighbors by raw
    MCES (literal nearest, no range tiers -- the winning config from the
    batch-composition simulations: consistently high enrichment of
    MCES<10 pairs with no batches landing near-zero close pairs), giving
    n = n_anchors * (k_nearest + 1) unique molecules total. Each molecule
    contributes 2 spectra (different indices when it has >=2 available),
    assembled and augmented ONCE each (not once per comparison) via
    CustomDatasetMultitasking._assemble_single_spectrum, then scored as
    the full (n, n) cross matrix by the model -- so the transformer
    encodes 2n spectra per batch, not the n**2 that a naive
    one-pair-at-a-time construction would require."""

    def __init__(
        self,
        dataset: CustomDatasetMultitasking,
        dense_mces: np.ndarray,
        n_anchors: int,
        k_nearest: int,
        mces_max_value: float,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.dense_mces = dense_mces
        self.n_anchors = n_anchors
        self.k_nearest = k_nearest
        self.n_molecules_per_batch = n_anchors * (k_nearest + 1)
        self.mces_max_value = mces_max_value
        self.n_mol = dense_mces.shape[0]
        self._seed = seed

    def __len__(self):
        # Lightning caps each epoch at min(training.limit_train_batches,
        # len(dataset)) -- NOT just limit_train_batches alone. __iter__
        # never stops on its own (infinite generator), so this must return
        # something >= any limit_train_batches value ever used, or a
        # smaller "coverage epoch" estimate here would silently override
        # the configured limit_train_batches and under-train per epoch
        # (confirmed: n_anchors=16 gave 1500 steps/epoch instead of the
        # configured 10000, since 24010//16=1500 < 10000).
        return 10**9

    def _sample_two_spectra(self, mol_idx: int, rng: random.Random) -> tuple[int, int]:
        indexes = self.dataset.df_smiles.loc[mol_idx, "indexes"]
        if len(indexes) >= 2:
            a, b = rng.sample(list(indexes), 2)
            return a, b
        only = rng.choice(list(indexes))
        return only, only

    def _sample_batch_molecules(self, np_rng: np.random.Generator) -> list[int]:
        target_n = self.n_molecules_per_batch
        selected: list[int] = []
        selected_mask = np.zeros(self.n_mol, dtype=bool)

        anchors = np_rng.choice(self.n_mol, size=self.n_anchors, replace=False)
        for a in anchors:
            if not selected_mask[a]:
                selected.append(int(a))
                selected_mask[a] = True

        for a in anchors:
            row = self.dense_mces[a].copy()
            row[selected_mask] = np.inf
            row[a] = np.inf
            k = self.k_nearest
            nearest = np.argpartition(row, k)[:k]
            nearest = nearest[np.argsort(row[nearest])]
            for c in nearest:
                if not selected_mask[c]:
                    selected.append(int(c))
                    selected_mask[c] = True

        if len(selected) < target_n:
            cand = np.where(~selected_mask)[0]
            extra = np_rng.choice(cand, size=target_n - len(selected), replace=False)
            selected.extend(int(x) for x in extra)
        return selected[:target_n]

    def _stack_view(self, specs: list[dict]) -> dict:
        return {
            key: torch.from_numpy(np.stack([s[key] for s in specs]))
            for key in ("mz", "intensity", "precursor_mass", "precursor_charge")
        }

    def __iter__(self):
        # Offset the seed per DataLoader worker -- otherwise every worker
        # would share the same seed and yield identical, duplicated
        # batches instead of independent random draws.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self._seed + worker_id)
        np_rng = np.random.default_rng(self._seed + worker_id)
        while True:
            mols = self._sample_batch_molecules(np_rng)
            view_a_specs, view_b_specs = [], []
            for mol_idx in mols:
                spec_a, spec_b = self._sample_two_spectra(mol_idx, rng)
                view_a_specs.append(
                    self.dataset._assemble_single_spectrum(mol_idx, spec_a)
                )
                view_b_specs.append(
                    self.dataset._assemble_single_spectrum(mol_idx, spec_b)
                )

            mols_arr = np.array(mols)
            raw_mces = self.dense_mces[np.ix_(mols_arr, mols_arr)].copy()
            np.fill_diagonal(raw_mces, 0.0)
            target = np.clip(1.0 - raw_mces / self.mces_max_value, 0.0, None)

            yield {
                "view_a": self._stack_view(view_a_specs),
                "view_b": self._stack_view(view_b_specs),
                "mces_target_matrix": torch.from_numpy(target.astype(np.float32)),
            }
