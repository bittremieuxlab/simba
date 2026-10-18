"""Analog discovery (paper Figure 2 reproduction): apply SIMBA.

Two subcommands, both needed by analog_discovery_plot.py:

  dedup        Embeds CASMI queries + one search's DEDUPLICATED library
               (one spectrum per molecule) with the paper checkpoint and
               scores every query x library pair. Produces
               `smiles_query.txt`/`smiles_library.txt` -- the canonical
               per-molecule library ordering everything else (ground-truth
               caches, baseline-model alignment) is keyed against.

  no_mol_agg   Embeds + scores EVERY library spectrum directly, duplicates
               allowed -- matching Sebastian's own
               AnalogDiscovery.get_k_candidates, which operates on the raw
               per-spectrum similarity matrix with no molecule-level
               deduplication anywhere. Produces the per-spectrum rank +
               lib_mol_idx arrays analog_discovery_plot.py actually uses
               for SIMBA's curve (a molecule with many replicate spectra
               can occupy multiple top-K slots, same as his code).

Both score a pair the same way:
  - pred_mces: raw MCES regression, (1 - cosine(emb0, emb1)) * MCES20_MAX_VALUE
               (same rescale EmbedderMultitask.compute_from_embeddings /
               Config.MCES20_MAX_VALUE=40 use at training time)
  - pred_edit_distance_class: argmax of the 6-way ordinal edit-distance head
               (used only as a MCES tiebreak, see
               simba/analog_discovery/paper_metrics.py::rank_by_mces_then_edit_distance)

Run once per search for each subcommand, on the MGF produced by
analog_discovery_prepare.py's build_mgf subcommand (search_A_nist_msg.mgf /
search_B_gnps.mgf; a FOLD=query/library field marks which half is which).

Usage:
    python tools/analog_discovery_score_simba.py dedup \\
        --checkpoint /mnt/data2/nkubrakov/paper_checkpoint/best_model_paper_SIMBA.ckpt \\
        --mgf /mnt/data2/nkubrakov/analog_discovery/search_A_nist_msg.mgf \\
        --output_dir /mnt/data2/nkubrakov/analog_discovery/scores/search_A

    python tools/analog_discovery_score_simba.py no_mol_agg \\
        --checkpoint /mnt/data2/nkubrakov/paper_checkpoint/best_model_paper_SIMBA.ckpt \\
        --mgf /mnt/data2/nkubrakov/analog_discovery/search_A_nist_msg.mgf \\
        --scores_dir /mnt/data2/nkubrakov/analog_discovery/scores/search_A \\
        --library a --training_pkl /mnt/data2/nkubrakov/preprocessing_ed_mces_20250123/edit_distance_neurips_nist_exhaustive.pkl \\
        --label "Search A (NIST20 + MassSpecGym)" \\
        --output_dir /mnt/data2/nkubrakov/analog_discovery/results_final_no_mol_agg

    # (repeat no_mol_agg for search B with --library b --gnps_mgf data/gnps/ALL_GNPS_NO_PROPOGATED.mgf)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simba.analog_discovery.paper_metrics import rank_by_mces_then_edit_distance
from simba.ordinal_classification.embedder_multitask import EmbedderMultitask
from simba.preprocessor import Preprocessor
from simba.spectrum_ext import SpectrumExt

MAX_NUM_PEAKS = 100  # Config.TRANSFORMER_CONTEXT -- must match training
MCES20_MAX_VALUE = 40.0  # Config.MCES20_MAX_VALUE
MIN_N_PEAKS = 6

# EmbedderMultitask constructor kwargs matching the paper's actual training
# command: d_model/n_layers/n_classes are Config defaults (D_MODEL=256,
# N_LAYERS=5, EDIT_DISTANCE_N_CLASSES=6); USE_LEARNABLE_MULTITASK and
# USE_EXTRA_METADATA_MODEL were explicitly overridden to 0/False on the
# paper run. Everything else left at Config() defaults.
MODEL_KWARGS = dict(
    d_model=256,
    n_layers=5,
    n_classes=6,
    use_gumbel=False,
    weights=None,
    lr=1e-4,
    use_cosine_distance=True,
    weights_sim2=None,
    use_edit_distance_regresion=False,
    use_mces20_log_loss=False,
    use_precursor_mz_for_model=True,
    use_extra_metadata=False,
    use_categorical_adducts=False,
    adduct_info_csv="",
    USE_LEARNABLE_MULTITASK=False,
)


def load_model(checkpoint: str, device: torch.device) -> EmbedderMultitask:
    model = EmbedderMultitask.load_from_checkpoint(
        checkpoint, map_location=device, strict=True, **MODEL_KWARGS
    )
    model.eval()
    model.to(device)
    return model


def _parse_mgf_fold(mgf_path: str, fold: str) -> list[dict]:
    blocks = []
    cur = None
    cur_fold = None
    with open(mgf_path, errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                cur = {"_mz": [], "_intensity": []}
                cur_fold = None
                continue
            if line == "END IONS":
                if cur is not None and cur_fold == fold:
                    blocks.append(cur)
                cur = None
                continue
            if cur is None:
                continue
            if "=" in line and not line[0].isdigit():
                key, _, val = line.partition("=")
                key = key.strip().upper()
                val = val.strip()
                cur[key] = val
                if key == "FOLD":
                    cur_fold = val
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        inten = float(parts[1])
                    except ValueError:
                        continue
                    cur["_mz"].append(mz)
                    cur["_intensity"].append(inten)
    return blocks


def _block_to_spectrum_ext(block: dict, idx: int) -> SpectrumExt:
    return SpectrumExt(
        identifier=str(idx),
        precursor_mz=float(block.get("PEPMASS", 0) or 0),
        precursor_charge=int(float(block.get("CHARGE", 1) or 1)),
        mz=np.asarray(block["_mz"], dtype=np.float64),
        intensity=np.asarray(block["_intensity"], dtype=np.float64),
        retention_time=0.0,
        params={"smiles": block.get("SMILES", ""), "adduct": block.get("ADDUCT", "")},
        library=None,
        inchi=None,
        smiles=block.get("SMILES", ""),
        ionmode="positive",
        bms=None,
        superclass=None,
        classe=None,
        subclass=None,
    )


def load_spectra(mgf_path: str, fold: str) -> tuple[list[str], list[SpectrumExt]]:
    blocks = _parse_mgf_fold(mgf_path, fold)
    # Defensive: remove_precursor_peak divides by precursor_mz internally and
    # raises ZeroDivisionError on a 0/missing value.
    n_before = len(blocks)
    blocks = [b for b in blocks if float(b.get("PEPMASS", 0) or 0) > 0]
    if len(blocks) != n_before:
        print(f"  WARNING: dropped {n_before - len(blocks)} '{fold}' spectra with missing/zero PEPMASS")
    spectra = [_block_to_spectrum_ext(b, i) for i, b in enumerate(blocks)]
    pp = Preprocessor()
    spectra = pp.preprocess_all_spectrums(spectra, max_num_peaks=MAX_NUM_PEAKS, training=False)
    smiles = [b.get("SMILES", "") for b in blocks]
    return smiles, spectra


@torch.no_grad()
def embed_spectra(model: EmbedderMultitask, spectra: list[SpectrumExt], batch_size: int, device: torch.device) -> torch.Tensor:
    embs = []
    for start in tqdm(range(0, len(spectra), batch_size), desc="Embedding", unit="batch"):
        chunk = spectra[start : start + batch_size]
        n = len(chunk)
        mz = np.zeros((n, MAX_NUM_PEAKS), dtype=np.float32)
        intensity = np.zeros((n, MAX_NUM_PEAKS), dtype=np.float32)
        precursor_mass = np.zeros((n, 1), dtype=np.float32)
        precursor_charge = np.zeros((n, 1), dtype=np.float32)
        for i, s in enumerate(chunk):
            length = min(len(s.mz), MAX_NUM_PEAKS)
            mz[i, :length] = np.asarray(s.mz[:length])
            intensity[i, :length] = np.asarray(s.intensity[:length])
            precursor_mass[i] = s.precursor_mz
            precursor_charge[i] = s.precursor_charge

        batch = {
            "mz_0": torch.from_numpy(mz).to(device),
            "intensity_0": torch.from_numpy(intensity).to(device),
            "precursor_mass_0": torch.from_numpy(precursor_mass).to(device),
            "precursor_charge_0": torch.from_numpy(precursor_charge).to(device),
        }
        kwargs = {
            "precursor_mass": batch["precursor_mass_0"].float(),
            "precursor_charge": batch["precursor_charge_0"].float(),
        }
        emb, _ = model.spectrum_encoder(
            mz_array=batch["mz_0"].float(),
            intensity_array=batch["intensity_0"].float(),
            **kwargs,
        )
        emb = emb[:, 0, :]
        emb = model.relu(emb)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def score_pairs(
    model: EmbedderMultitask,
    q_embs: torch.Tensor,
    lib_embs: torch.Tensor,
    device: torch.device,
    pair_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_q, n_lib = q_embs.shape[0], lib_embs.shape[0]
    pred_mces = np.empty((n_q, n_lib), dtype=np.float32)
    pred_edit_class = np.empty((n_q, n_lib), dtype=np.int32)

    for i in tqdm(range(n_q), desc="SIMBA pairwise scoring", unit="query"):
        emb0_full = q_embs[i : i + 1]
        for start in range(0, n_lib, pair_chunk_size):
            end = min(start + pair_chunk_size, n_lib)
            n_chunk = end - start
            emb0 = emb0_full.expand(n_chunk, -1).to(device)
            emb1 = lib_embs[start:end].to(device)

            edit_logits, cos_sim = model.compute_from_embeddings(emb0, emb1)
            pred_mces[i, start:end] = ((1.0 - cos_sim) * MCES20_MAX_VALUE).cpu().numpy()
            pred_edit_class[i, start:end] = edit_logits.argmax(dim=-1).cpu().numpy()

    return pred_mces, pred_edit_class


# ══════════════════════════════════════════════════════════════════════════
# dedup -- one spectrum per molecule (canonical library ordering)
# ══════════════════════════════════════════════════════════════════════════


def run_dedup(checkpoint: str, mgf: str, output_dir: str, batch_size: int = 256, pair_chunk_size: int = 20000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint} (device={device}) ...")
    model = load_model(checkpoint, device)

    print(f"Loading query/library spectra from {mgf} ...")
    q_smiles, q_spectra = load_spectra(mgf, "query")
    lib_smiles, lib_spectra = load_spectra(mgf, "library")
    print(f"  {len(q_smiles)} query molecules, {len(lib_smiles)} library molecules")

    print("Embedding queries ...")
    q_embs = embed_spectra(model, q_spectra, batch_size, device)
    print("Embedding library ...")
    lib_embs = embed_spectra(model, lib_spectra, batch_size, device)

    print("Scoring SIMBA (MCES regression + edit-distance tiebreak class) ...")
    pred_mces, pred_edit_class = score_pairs(model, q_embs, lib_embs, device, pair_chunk_size)

    print(f"Saving to {out_dir} ...")
    out_dir.joinpath("smiles_query.txt").write_text("\n".join(q_smiles) + "\n")
    out_dir.joinpath("smiles_library.txt").write_text("\n".join(lib_smiles) + "\n")
    np.save(out_dir / "score_simba_mces.npy", pred_mces)
    np.save(out_dir / "score_simba_edit_distance_class.npy", pred_edit_class)

    print("Done.")


# ══════════════════════════════════════════════════════════════════════════
# no_mol_agg -- every library spectrum, duplicates allowed
# ══════════════════════════════════════════════════════════════════════════


def load_library_a_all(training_pkl: str, lib_pos: dict) -> tuple[list[str], list]:
    """Library A (NIST20+MassSpecGym): every retained spectrum (not just one
    per molecule) from SIMBA's own training pickle, restricted to molecules
    already present in the dedup library ordering (lib_pos)."""
    import sys as _sys

    import simba

    _sys.modules["src"] = simba
    import dill

    from tools.analog_discovery_prepare import canon

    with open(training_pkl, "rb") as f:
        dataset = dill.load(f)
    exact_members = set()
    for split in ("molecule_pairs_train", "molecule_pairs_val", "molecule_pairs_test"):
        for s in dataset[split].spectrums:
            c = canon(s.smiles)
            if c:
                exact_members.add(c)
    lib_smiles_all, lib_spectra_all = [], []
    for split in ("molecule_pairs_train", "molecule_pairs_val", "molecule_pairs_test"):
        for s in dataset[split].spectrums_original:
            if len(s.mz) < MIN_N_PEAKS:
                continue
            c = canon(s.smiles)
            if c is None or c not in exact_members or c not in lib_pos:
                continue
            lib_smiles_all.append(c)
            lib_spectra_all.append(s)
    return lib_smiles_all, lib_spectra_all


def load_library_b_all(gnps_mgf: str, lib_pos: dict) -> tuple[list[str], list[SpectrumExt]]:
    """Library B (GNPS): every retained spectrum (not just one per
    molecule) from the raw GNPS MGF, restricted to molecules already
    present in the dedup library ordering (lib_pos)."""
    from tools.analog_discovery_prepare import canon

    lib_smiles_all, lib_spectra_all = [], []
    cur, n_seen = None, 0
    with open(gnps_mgf, errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                cur = {"_mz": [], "_intensity": []}
                continue
            if line == "END IONS":
                if cur is not None:
                    n_seen += 1
                    smi = canon(cur.get("SMILES"))
                    ionmode = (cur.get("IONMODE") or "").strip().lower()
                    try:
                        precursor_mz = float(cur.get("PEPMASS", 0) or 0)
                    except ValueError:
                        precursor_mz = 0.0
                    if (smi and smi in lib_pos and len(cur["_mz"]) >= MIN_N_PEAKS
                            and ionmode in ("", "positive") and precursor_mz > 0):
                        m = max(cur["_intensity"]) if cur["_intensity"] else 0.0
                        inten = [i / m for i in cur["_intensity"]] if m > 0 else cur["_intensity"]
                        lib_smiles_all.append(smi)
                        lib_spectra_all.append(SpectrumExt(
                            identifier=str(n_seen), precursor_mz=precursor_mz, precursor_charge=1,
                            mz=np.asarray(cur["_mz"], dtype=np.float64), intensity=np.asarray(inten, dtype=np.float64),
                            retention_time=0.0, params={"smiles": smi}, library=None, inchi=None, smiles=smi,
                            ionmode="positive", bms=None, superclass=None, classe=None, subclass=None,
                        ))
                cur = None
                continue
            if cur is None:
                continue
            if "=" in line and not line[0].isdigit():
                key, _, val = line.partition("=")
                cur[key.strip().upper()] = val.strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz, inten = float(parts[0]), float(parts[1])
                    except ValueError:
                        continue
                    cur["_mz"].append(mz)
                    cur["_intensity"].append(inten)
    return lib_smiles_all, lib_spectra_all


def run_no_mol_agg(
    checkpoint: str,
    mgf: str,
    scores_dir: str,
    library: str,
    label: str,
    output_dir: str,
    training_pkl: str | None = None,
    gnps_mgf: str | None = None,
    batch_size: int = 256,
    pair_chunk_size: int = 20000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint} (device={device}) ...")
    model = load_model(checkpoint, device)

    print(f"Loading query spectra from {mgf} ...")
    q_smiles, q_spectra = load_spectra(mgf, "query")
    print(f"  {len(q_smiles)} queries")

    lib_smiles_dedup = Path(scores_dir, "smiles_library.txt").read_text().splitlines()
    lib_pos = {s: i for i, s in enumerate(lib_smiles_dedup)}

    if library == "a":
        if not training_pkl:
            raise SystemExit("--training_pkl is required for --library a")
        lib_smiles_all, lib_spectra_all = load_library_a_all(training_pkl, lib_pos)
    elif library == "b":
        if not gnps_mgf:
            raise SystemExit("--gnps_mgf is required for --library b")
        lib_smiles_all, lib_spectra_all = load_library_b_all(gnps_mgf, lib_pos)
    else:
        raise SystemExit(f"--library must be 'a' or 'b', got {library!r}")
    print(f"  {len(lib_spectra_all)} spectra across {len(set(lib_smiles_all))} unique molecules (no dedup)")
    lib_mol_idx = np.array([lib_pos[s] for s in lib_smiles_all], dtype=np.int64)

    print("Embedding queries ...")
    q_embs = embed_spectra(model, q_spectra, batch_size, device)
    print("Embedding library (every spectrum) ...")
    lib_embs = embed_spectra(model, lib_spectra_all, batch_size, device)

    print("Scoring SIMBA (MCES regression + edit-distance tiebreak class) ...")
    pred_mces, pred_edit = score_pairs(model, q_embs, lib_embs, device, pair_chunk_size)

    rank = rank_by_mces_then_edit_distance(pred_mces, pred_edit)  # (n_q, n_lib_spec), 0=best

    print(f"Saving to {out_dir} ...")
    np.save(out_dir / f"{label}_rank.npy", rank)
    np.save(out_dir / f"{label}_lib_mol_idx.npy", lib_mol_idx)
    print("Done.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("dedup", help="Score the deduplicated library (one spectrum/molecule).")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--mgf", required=True, help="Combined query+library MGF with FOLD= field")
    s.add_argument("--output_dir", required=True)
    s.add_argument("--batch_size", type=int, default=256)
    s.add_argument("--pair_chunk_size", type=int, default=20000)

    s = sub.add_parser("no_mol_agg", help="Score every library spectrum, duplicates allowed.")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--mgf", required=True, help="Combined query+library MGF with FOLD= field (for queries)")
    s.add_argument("--scores_dir", required=True, help="dedup subcommand's output_dir (for smiles_library.txt ordering)")
    s.add_argument("--library", required=True, choices=["a", "b"])
    s.add_argument("--training_pkl", default=None, help="required for --library a")
    s.add_argument("--gnps_mgf", default=None, help="required for --library b")
    s.add_argument("--label", required=True, help='e.g. "Search A (NIST20 + MassSpecGym)" -- output file prefix')
    s.add_argument("--output_dir", required=True)
    s.add_argument("--batch_size", type=int, default=256)
    s.add_argument("--pair_chunk_size", type=int, default=20000)

    args = p.parse_args()
    kwargs = vars(args)
    cmd = kwargs.pop("cmd")
    if cmd == "dedup":
        run_dedup(**kwargs)
    elif cmd == "no_mol_agg":
        run_no_mol_agg(**kwargs)


if __name__ == "__main__":
    main()
