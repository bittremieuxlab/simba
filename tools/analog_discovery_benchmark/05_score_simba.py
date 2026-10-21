"""Step 5: SIMBA scores for Search A and B, literally following Sebastian's
notebooks/analog_discovery_casmi_msnlib.ipynb (simba_v1):

  Encoder(checkpoint, multitasking=True) embeds queries and library spectra
  via LoadDataEncoder.from_spectrums_to_dataset (SIMBA preprocessing + L2-
  normalised intensities); FcLayerAnalogDiscovery.compute_all_combinations
  gives, per pair, SIMBA's predicted MCES similarity sim_mces = 1 - MCES/40
  (raw MCES distance in bonds, rescaled as in training; NOT the normalized
  MCES used as ground truth) and edit-distance logits. compute_ranking orders
  each query's library by round(sim_mces * 40) = 40 - predicted MCES in whole
  bonds, then the predicted edit-distance class, then sim_mces (all
  descending). The resulting ranking matrix (higher = better) is what his
  notebooks rank with.

Needs a GPU (embedding). Output in <out_dir>/search_{A,B}/:
  ranking.npy (float32), sim_mces.npy (float32), ed_class.npy (uint8),
  smiles_query.txt, smiles_library.txt
"""

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from casmi_sets import SEARCHES, SetLoader, canonical, check_order
from simba.analog_discovery.fc_layers_analog_discovery import FcLayerAnalogDiscovery
from simba.transformers.encoder import Encoder
from simba.transformers.load_data_encoder import LoadDataEncoder


def compute_ranking(similarities1, similarities1_integer, similarities2_integer):
    """His compute_ranking: lexsort by (integer MCES, edit-distance class, MCES), descending."""
    ranking_total = np.zeros(similarities1.shape, dtype=float)
    for row_index, (row_sim, row_int, row_int2) in enumerate(zip(similarities1, similarities1_integer, similarities2_integer)):
        sorted_indices = np.lexsort((-row_sim, -row_int2, -row_int))
        ranking = np.empty_like(sorted_indices, dtype=float)
        ranking[sorted_indices] = np.arange(len(row_sim), 0, -1)
        ranking /= len(row_sim)
        ranking_total[row_index] = ranking
    return ranking_total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="SIMBA multitask checkpoint (.ckpt)")
    ap.add_argument("--dataset_pkl", required=True)
    ap.add_argument("--casmi_pkl", required=True)
    ap.add_argument("--gnps_mgf", required=True)
    ap.add_argument("--sets_dir", required=True, help="output of 01_build_sets.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--searches", nargs="+", default=list(SEARCHES), choices=SEARCHES)
    args = ap.parse_args()

    loader = SetLoader(args.dataset_pkl, args.casmi_pkl, args.gnps_mgf)
    config = loader.config  # his notebook settings
    config.D_MODEL, config.N_LAYERS, config.TRANSFORMER_CONTEXT, config.USE_FINGERPRINT = 256, 5, 100, 0
    config.use_cosine_distance = True
    encoder = Encoder(args.checkpoint, D_MODEL=int(config.D_MODEL), N_LAYERS=int(config.N_LAYERS), multitasking=True, config=config)

    def embed(spectra):
        dataset = LoadDataEncoder.from_spectrums_to_dataset(spectra, max_num_peaks=int(config.TRANSFORMER_CONTEXT))
        return encoder.get_embeddings(DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=0))

    for search in args.searches:
        queries, library = loader.load(search)
        check_order(args.sets_dir, search, queries, library)
        emb_q, emb_lib = embed(queries), embed(library)
        print(f"Search {search}: embedded {emb_q.shape} queries, {emb_lib.shape} library spectra", flush=True)
        # his naming: similarities2 = edit-distance logits, similarities1 = MCES similarity
        similarities2, similarities1 = FcLayerAnalogDiscovery.compute_all_combinations(args.checkpoint, emb_q, emb_lib, config)
        ed_class = np.argmax(similarities2, axis=2)
        del similarities2
        ranking = compute_ranking(similarities1, np.round(similarities1 * 40), ed_class)

        out = Path(args.out_dir) / f"search_{search}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "ranking.npy", ranking.astype(np.float32))
        np.save(out / "sim_mces.npy", similarities1.astype(np.float32))
        np.save(out / "ed_class.npy", ed_class.astype(np.uint8))
        out.joinpath("smiles_query.txt").write_text("\n".join(canonical(queries)) + "\n")
        out.joinpath("smiles_library.txt").write_text("\n".join(canonical(library)) + "\n")
        print(f"Search {search}: ranking {ranking.shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()
