"""Analog discovery (paper Figure 2 reproduction): prepare all data +
ground truth that analog_discovery_plot.py needs (everything upstream of
applying SIMBA itself, see analog_discovery_score_simba.py for that).

One file, seven subcommands, run in this order:

  build_mgf              CASMI + GNPS -> the two search MGFs (query +
                         library spectra, FOLD=query/library tagged).
  exact_mces_prepare     \\
  exact_mces_compute_block |  exhaustive EXACT raw MCES (myopic_mces ILP,
  exact_mces_status        |  threshold=20) over one search's full query x
  exact_mces_combine    /   library pair space -- same block-split design
                            as before, just folded into this one file.
  normalize_shard        \\  NORMALIZED MCES via RASCAL -- both metrics
  normalize_combine      /  supported via --metric/--cache_name (sebastian
                            = paper's actual tier-similarity formula,
                            default; johnson = paper Methods text's stated
                            formula) -- sharded the same way, folded into
                            this one file. Run normalize_shard/combine
                            TWICE per search, once per metric.

Each of the two exhaustive, millions-of-pairs steps (exact_mces_*,
normalize_*) is still split into independent blocks/shards -- that's not
extra complexity for its own sake, it's what makes either one affordable
at all (a single process doing this serially would take days); split
--task_id/--shard_id across as many parallel workers/machines as you have.
Nothing here needs SLURM -- call these directly, however many at a time
your hardware allows.

Usage: see each subcommand's --help, or tools/ANALOG_DISCOVERY_README.md
for a full worked example end to end.
"""

import argparse
import json
import multiprocessing
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simba.analog_discovery.paper_metrics import (
    normalized_mces_distance_johnson,
    normalized_mces_distance_sebastian,
)

_NORMALIZE_METRIC_FNS = {
    "sebastian": normalized_mces_distance_sebastian,
    "johnson": normalized_mces_distance_johnson,
}

MIN_N_PEAKS = 6
EXACT_MCES_THRESHOLD = 20
BAILOUT_RAW_THRESHOLD = 5
BAILOUT_NORM_THRESHOLD = 0.5


def canon(smi: str | None) -> str | None:
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# ══════════════════════════════════════════════════════════════════════════
# build_mgf -- CASMI + GNPS -> the two search MGFs
# ══════════════════════════════════════════════════════════════════════════
#
# METHODOLOGY (confirmed exact match to the paper's reported counts):
#
# 1. CASMI queries: restrict to the M+H (singly protonated) adduct only,
#    then run the real SIMBA preprocessing pipeline (remove_precursor_peak,
#    filter_intensity min 1% + cap 100 peaks) and keep only spectra with
#    >= 6 peaks remaining. Reproduces the paper's stated starting count
#    exactly: 170 M+H spectra -> 158 after this filtering.
# 2. Reference library for exclusion: the paper's library A is the literal
#    42,774-unique-compound set actually used for SIMBA's own
#    train/val/test pairs (edit_distance_neurips_nist_exhaustive.pkl's
#    molecule_pairs_{train,val,test}.spectrums) -- not an approximation
#    from raw NIST20/MassSpecGym scrapes. Using this exact set as the
#    exclusion criterion reproduces 158 -> 132 exactly.
# 3. GNPS is NOT used for query exclusion, only as Search B's search
#    target. The SAME 132-molecule query set is used for both searches.
#
# Data sources:
#   CASMI query data : casmi2022_spec_df.pkl (official paper data, Zenodo 15275257)
#   Library A (exact) : edit_distance_neurips_nist_exhaustive.pkl (the real training pickle)
#   GNPS              : ALL_GNPS_NO_PROPOGATED.mgf (https://external.gnps2.org/gnpslibrary/)


def normalize_intensity(intensity: list[float]) -> list[float]:
    """Rescale to relative-to-base-peak (max=1.0), matching the convention
    the model's training/library data already uses. CASMI and GNPS store
    raw absolute instrument intensities (up to 1e7+) instead -- feeding
    those unnormalized into the model, whose intensity positional encoding
    only spans wavelengths 1e-6 to 1, silently produces near-meaningless
    embeddings."""
    m = max(intensity) if intensity else 0.0
    if m <= 0:
        return intensity
    return [i / m for i in intensity]


def _write_mgf_entry(fh, smiles: str, adduct: str, precursor_mz: float, charge: int, mz, intensity, fold: str):
    fh.write("BEGIN IONS\n")
    fh.write(f"SMILES={smiles}\n")
    fh.write(f"ADDUCT={adduct}\n")
    fh.write(f"PEPMASS={precursor_mz}\n")
    fh.write(f"CHARGE={charge}\n")
    fh.write("IONMODE=positive\n")
    fh.write(f"FOLD={fold}\n")
    for m, i in zip(mz, intensity):
        fh.write(f"{m} {i}\n")
    fh.write("END IONS\n")


def load_casmi_queries(casmi_pkl: str) -> dict[str, dict]:
    import simba

    sys.modules["src"] = simba
    from simba.preprocessor import Preprocessor
    from simba.spectrum_ext import SpectrumExt

    with open(casmi_pkl, "rb") as f:
        df = pickle.load(f)
    mh = df[df["prec_type"] == "[M+H]+"]
    print(f"  CASMI2022: {len(df)} total rows -> {len(mh)} M+H-adduct rows")

    raw_entries = {}
    spectra_for_filter = []
    order = []
    for _, row in mh.iterrows():
        smi = canon(row["smiles"])
        if smi is None or smi in raw_entries:
            continue
        peaks = row["peaks"]
        mz = [p[0] for p in peaks]
        intensity = normalize_intensity([p[1] for p in peaks])
        raw_entries[smi] = {
            "mz": mz, "intensity": intensity, "adduct": row["prec_type"],
            "precursor_mz": row["prec_mz"], "charge": 1,
        }
        spectra_for_filter.append(
            SpectrumExt(
                identifier=smi, precursor_mz=row["prec_mz"], precursor_charge=1,
                mz=np.asarray(mz, dtype=np.float64), intensity=np.asarray(intensity, dtype=np.float64),
                retention_time=0.0, params={"smiles": smi}, library=None, inchi=None, smiles=smi,
                ionmode="positive", bms=None, superclass=None, classe=None, subclass=None,
            )
        )
        order.append(smi)

    pp = Preprocessor()
    processed = pp.preprocess_all_spectrums(list(spectra_for_filter), max_num_peaks=100, training=False)
    survivors = {smi for smi, s in zip(order, processed) if len(s.mz) >= MIN_N_PEAKS}

    out = {smi: raw_entries[smi] for smi in survivors}
    print(f"  -> {len(raw_entries)} unique M+H molecules -> {len(out)} survive >= {MIN_N_PEAKS}-peak filter "
          f"after real preprocessing (expected: 158 spectra / ~157 unique)")
    return out


def load_library_a_exact(training_pkl: str) -> dict[str, dict]:
    """The exact set of molecules SIMBA was actually trained/validated/
    tested on (not an approximation from raw library pickles). NOTE:
    dataset[split].spectrums is per-unique-molecule bookkeeping with dummy
    placeholder peaks (mz=[0.]) -- real peak data lives in
    dataset[split].spectrums_original. .spectrums is used only to decide
    exact set membership; peaks are pulled from .spectrums_original."""
    import simba

    sys.modules["src"] = simba
    import dill

    with open(training_pkl, "rb") as f:
        dataset = dill.load(f)

    exact_members = set()
    n_total = 0
    for split in ("molecule_pairs_train", "molecule_pairs_val", "molecule_pairs_test"):
        spectrums = dataset[split].spectrums
        n_total += len(spectrums)
        for s in spectrums:
            smi = canon(s.smiles)
            if smi is not None:
                exact_members.add(smi)

    out = {}
    for split in ("molecule_pairs_train", "molecule_pairs_val", "molecule_pairs_test"):
        for s in dataset[split].spectrums_original:
            if len(s.mz) < MIN_N_PEAKS:
                continue
            smi = canon(s.smiles)
            if smi is None or smi not in exact_members or smi in out:
                continue
            out[smi] = {
                "mz": list(s.mz), "intensity": list(s.intensity), "adduct": "[M+H]+",
                "precursor_mz": s.precursor_mz, "charge": 1,
            }
    print(f"  Library A (exact training data): {n_total} spectra across train/val/test -> {len(exact_members)} unique "
          f"molecules (expected: 42,774), {len(out)} with real peak data recovered from spectrums_original")
    return out


def load_gnps(gnps_mgf: str) -> dict[str, dict]:
    out = {}
    n_seen = 0
    cur = None
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
                    if (
                        smi and smi not in out and len(cur["_mz"]) >= MIN_N_PEAKS
                        and ionmode in ("", "positive") and precursor_mz > 0
                    ):
                        out[smi] = {
                            "mz": cur["_mz"], "intensity": normalize_intensity(cur["_intensity"]),
                            "adduct": cur.get("ADDUCT") or "[M+H]+", "precursor_mz": precursor_mz, "charge": 1,
                        }
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
    print(f"  GNPS: {n_seen} spectra -> {len(out)} unique canonical SMILES (positive mode, >= {MIN_N_PEAKS} peaks)")
    return out


def build_mgf(casmi_pkl: str, training_pkl: str, gnps_mgf: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CASMI2022 queries (M+H-only, real-preprocessing-filtered) ...")
    casmi = load_casmi_queries(casmi_pkl)
    print("\nLoading Library A (exact training data) ...")
    library_a = load_library_a_exact(training_pkl)
    print("\nLoading GNPS (Search B's library -- not used for query exclusion) ...")
    library_b = load_gnps(gnps_mgf)
    print(f"Library B (GNPS): {len(library_b)} unique molecules")

    queries = {smi: b for smi, b in casmi.items() if smi not in library_a}
    print(
        f"\nCASMI queries: {len(casmi)} clean -> {len(queries)} remain after excluding molecules present in "
        f"Library A only (excluded {len(casmi) - len(queries)}; paper reports 132 remaining out of 158)."
    )

    search_a_path = out_dir / "search_A_nist_msg.mgf"
    search_b_path = out_dir / "search_B_gnps.mgf"

    print(f"\nWriting {search_a_path} ...")
    with open(search_a_path, "w") as fh:
        for smi, b in queries.items():
            _write_mgf_entry(fh, smi, b["adduct"], b["precursor_mz"], b["charge"], b["mz"], b["intensity"], "query")
        for smi, b in library_a.items():
            _write_mgf_entry(fh, smi, b["adduct"], b["precursor_mz"], b["charge"], b["mz"], b["intensity"], "library")

    print(f"Writing {search_b_path} ...")
    with open(search_b_path, "w") as fh:
        for smi, b in queries.items():
            _write_mgf_entry(fh, smi, b["adduct"], b["precursor_mz"], b["charge"], b["mz"], b["intensity"], "query")
        for smi, b in library_b.items():
            _write_mgf_entry(fh, smi, b["adduct"], b["precursor_mz"], b["charge"], b["mz"], b["intensity"], "library")

    print("\nDone.")


# ══════════════════════════════════════════════════════════════════════════
# exact_mces_* -- exhaustive EXACT raw MCES via myopic_mces ILP solver
# ══════════════════════════════════════════════════════════════════════════
#
# Block-split (prepare -> compute_block x N -> status -> combine) because
# this is exhaustive over every query x library pair (millions) and each
# solve can occasionally hang -- a watchdog kills+restarts the worker pool
# on a stall rather than blocking the whole block forever.


def _load_fold_smiles(mgf_path: str, fold: str) -> set[str]:
    smis = set()
    cur_fold = None
    cur_smi = None
    with open(mgf_path) as fh:
        for line in fh:
            line = line.strip()
            if line == "BEGIN IONS":
                cur_fold, cur_smi = None, None
            elif line.upper().startswith("FOLD="):
                cur_fold = line.split("=", 1)[1].strip()
            elif line.upper().startswith("SMILES="):
                cur_smi = line.split("=", 1)[1].strip()
            elif line == "END IONS" and cur_fold == fold and cur_smi:
                smis.add(cur_smi)
    return smis


def exact_mces_prepare(output_dir: Path, mgf: str, n_blocks: int) -> None:
    """Build the exhaustive query x library molecule-pair list for one search."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blocks").mkdir(exist_ok=True)

    print(f"Scanning {mgf} for query/library SMILES ...")
    query_canon = sorted({c for s in _load_fold_smiles(mgf, "query") if (c := canon(s))})
    library_canon = sorted({c for s in _load_fold_smiles(mgf, "library") if (c := canon(s))})
    print(f"  {len(query_canon)} query molecules, {len(library_canon)} library molecules")

    overlap = set(query_canon) & set(library_canon)
    if overlap:
        print(f"  WARNING: {len(overlap)} molecules present in BOTH query and library "
              "-- expected to be zero after build_mgf's per-search CASMI exclusion.")

    smiles_list = list(query_canon)
    mol_to_idx = {s: i for i, s in enumerate(smiles_list)}
    for s in library_canon:
        if s not in mol_to_idx:
            mol_to_idx[s] = len(smiles_list)
            smiles_list.append(s)

    query_idx = np.asarray([mol_to_idx[s] for s in query_canon], dtype=np.int32)
    library_idx = np.asarray([mol_to_idx[s] for s in library_canon], dtype=np.int32)

    print("Building exhaustive pair list (query x library) ...")
    pairs = np.empty((len(query_idx) * len(library_idx), 2), dtype=np.int32)
    pairs[:, 0] = np.repeat(query_idx, len(library_idx))
    pairs[:, 1] = np.tile(library_idx, len(query_idx))
    n_pairs = len(pairs)
    print(f"  {n_pairs:,} pairs")

    # Shuffle so blocks have similar difficulty distribution. Fixed seed.
    rng = np.random.default_rng(seed=42)
    rng.shuffle(pairs)

    smiles_path = output_dir / "smiles.txt"
    smiles_path.write_text("\n".join(smiles_list) + "\n")
    pairs_path = output_dir / "pairs.npy"
    np.save(pairs_path, pairs)

    bounds = [int(round(i * n_pairs / n_blocks)) for i in range(n_blocks + 1)]
    bounds[-1] = n_pairs

    meta = {
        "n_pairs": n_pairs, "n_mols": len(smiles_list), "n_query": len(query_idx),
        "n_library": len(library_idx), "threshold": EXACT_MCES_THRESHOLD, "n_blocks": n_blocks,
        "bounds": bounds, "pairs_path": str(pairs_path.resolve()), "smiles_path": str(smiles_path.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {smiles_path}, {pairs_path}, meta.json -- {n_blocks} blocks ready.")


def _mces_worker(args):
    smi_a, smi_b, threshold = args
    from myopic_mces.myopic_mces import MCES as MCES2

    try:
        result = MCES2(
            smi_a, smi_b, threshold=threshold, i=0, solver="PULP_CBC_CMD",
            solver_options={"threads": 1, "msg": False, "timeLimit": 10},
            no_ilp_threshold=False, always_stronger_bound=True, catch_errors=True,
        )
        return float(result[1])
    except Exception:
        return -1.0


def _dispatch_with_watchdog(worker_args, n_jobs, per_pair_timeout, show_progress):
    """A pair stuck inside the ILP solver's native code never returns
    control to Python, so the only reliable way to bound per-pair wall
    time is to kill the whole worker process and resume the remaining
    pairs in a fresh pool (a signal-based timeout doesn't work here, same
    class of issue as RASCAL's own timeout not being preemptible)."""
    from tqdm.auto import tqdm

    n = len(worker_args)
    results = [None] * n
    todo = list(range(n))
    bar = tqdm(total=n, disable=not show_progress, unit="pair")

    while todo:
        pool = multiprocessing.Pool(n_jobs)
        it = pool.imap(_mces_worker, (worker_args[i] for i in todo))
        stuck_at = None
        try:
            for pos, idx in enumerate(todo):
                try:
                    results[idx] = it.next(timeout=per_pair_timeout)
                except multiprocessing.TimeoutError:
                    stuck_at = pos
                    break
                bar.update(1)
        finally:
            if stuck_at is not None:
                pool.terminate()
                pool.join()
                results[todo[stuck_at]] = -1.0
                bar.update(1)
                todo = todo[stuck_at + 1:]
            else:
                pool.close()
                pool.join()
                todo = []

    bar.close()
    return results


def exact_mces_compute_block(output_dir: Path, task_id: int, n_jobs: int, timeout: float | None) -> None:
    meta = json.loads((output_dir / "meta.json").read_text())
    n_blocks = meta["n_blocks"]

    if task_id >= n_blocks:
        print(f"task_id {task_id} >= n_blocks {n_blocks}; nothing to do.")
        return

    done_file = output_dir / "blocks" / f"block_{task_id:05d}.done"
    out_npy = output_dir / "blocks" / f"block_{task_id:05d}.npy"
    if done_file.exists():
        print(f"Block {task_id}: already done. Skipping.")
        return

    bounds = meta["bounds"]
    i0, i1 = bounds[task_id], bounds[task_id + 1]
    n_pair = i1 - i0
    print(f"Block {task_id}: pairs [{i0:,}, {i1:,})  --  {n_pair:,} pairs")

    smiles = Path(meta["smiles_path"]).read_text().splitlines()
    pairs = np.load(meta["pairs_path"], mmap_mode="r")[i0:i1]
    smiles_a = [smiles[i] for i in pairs[:, 0]]
    smiles_b = [smiles[j] for j in pairs[:, 1]]

    if n_jobs <= 0:
        # NEVER default to the whole host's core count -- see
        # analog_discovery_plot.py's build_normalized_gt_matrix for the
        # incident this guards against.
        if os.environ.get("SLURM_CPUS_PER_TASK"):
            n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])
        else:
            n_jobs = len(os.sched_getaffinity(0))
    per_pair_timeout = timeout if timeout else 300.0
    print(f"  MCES(threshold={meta['threshold']}, PULP_CBC_CMD, always_stronger_bound=True,"
          f" n_jobs={n_jobs}, per-pair timeout={per_pair_timeout}s, pool-restart watchdog)")

    worker_args = [(a, b, meta["threshold"]) for a, b in zip(smiles_a, smiles_b)]
    result = np.asarray(_dispatch_with_watchdog(worker_args, n_jobs, per_pair_timeout, show_progress=True), dtype=np.float32)

    n_failed = int((result == -1.0).sum() + np.isnan(result).sum())
    if n_failed:
        print(f"  {n_failed:,} / {n_pair:,} pairs failed to solve or timed out -- recorded as -1.")
    result = np.where(np.isnan(result), -1.0, result).astype(np.float32)

    np.save(out_npy, result)
    done_file.write_text(f"block={task_id} n_pairs={n_pair} n_failed={n_failed}\n")
    print(f"Block {task_id}: wrote {out_npy.name}. Done.")


def exact_mces_status(output_dir: Path) -> None:
    meta_path = output_dir / "meta.json"
    if not meta_path.exists():
        print(f"No meta.json at {output_dir}. Run exact_mces_prepare first.")
        return

    meta = json.loads(meta_path.read_text())
    n_blocks = meta["n_blocks"]
    bounds = meta["bounds"]
    done_ids = [i for i in range(n_blocks) if (output_dir / "blocks" / f"block_{i:05d}.done").exists()]
    pending_ids = [i for i in range(n_blocks) if i not in set(done_ids)]
    pairs_done = sum(bounds[i + 1] - bounds[i] for i in done_ids)
    n_pairs = meta["n_pairs"]

    print(f"Blocks : {len(done_ids):4d} / {n_blocks} done ({100 * len(done_ids) / n_blocks:.1f}%)")
    print(f"Pairs  : {pairs_done:,} / {n_pairs:,} ({100 * pairs_done / n_pairs:.1f}%)")
    if pending_ids:
        sample = pending_ids[:15]
        suffix = f" ... (+{len(pending_ids) - 15} more)" if len(pending_ids) > 15 else ""
        print(f"Pending: {sample}{suffix}")
    else:
        print("All blocks complete. Run: ... exact_mces_combine")


def exact_mces_combine(output_dir: Path) -> None:
    meta = json.loads((output_dir / "meta.json").read_text())
    n_blocks = meta["n_blocks"]

    missing = [i for i in range(n_blocks) if not (output_dir / "blocks" / f"block_{i:05d}.done").exists()]
    if missing:
        print(f"WARNING: {len(missing)} blocks not done: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    print(f"Loading {n_blocks} block files ...")
    chunks = []
    for i in range(n_blocks):
        p = output_dir / "blocks" / f"block_{i:05d}.npy"
        if p.exists():
            chunks.append(np.load(p))
        else:
            bounds = meta["bounds"]
            n_pair = bounds[i + 1] - bounds[i]
            print(f"  Block {i} missing -- filling with NaN ({n_pair:,} pairs)")
            chunks.append(np.full(n_pair, np.nan, dtype=np.float32))

    mces_values = np.concatenate(chunks).astype(np.float32)
    pairs = np.load(meta["pairs_path"])
    result = np.column_stack([pairs.astype(np.float32), mces_values])
    out_path = output_dir / "mces_exact.npy"
    np.save(out_path, result)
    print(f"Wrote {out_path}  shape={result.shape}")

    n_solver_failed = int((mces_values == -1.0).sum())
    valid = ~np.isnan(mces_values) & (mces_values != -1.0)
    print(f"Valid  : {valid.sum():,} / {len(mces_values):,}")
    if n_solver_failed:
        print(f"Failed : {n_solver_failed:,} pairs recorded as -1 (solver could not solve)")


# ══════════════════════════════════════════════════════════════════════════
# normalize_shard / normalize_combine -- NORMALIZED MCES via RASCAL
# ══════════════════════════════════════════════════════════════════════════
#
# Sharded the same way as exact_mces_* and for the same reason: millions of
# pairs, embarrassingly parallel, each pair cheap but a rare pathological
# molecule pair can be slow/memory-hungry (see paper_metrics.py's own
# guards) -- a shard is just "every Nth pair", no shared state between
# shards, so any number of them can run concurrently anywhere.


def load_scores_smiles(scores_dir: Path) -> tuple[list, list]:
    q_smiles = (scores_dir / "smiles_query.txt").read_text().splitlines()
    lib_smiles = (scores_dir / "smiles_library.txt").read_text().splitlines()
    return q_smiles, lib_smiles


def build_raw_gt_matrix(q_smiles: list, lib_smiles: list, exact_mces_dir: Path) -> np.ndarray:
    exact_smiles = (exact_mces_dir / "smiles.txt").read_text().splitlines()
    arr = np.load(exact_mces_dir / "mces_exact.npy")

    q_pos = {s: i for i, s in enumerate(q_smiles)}
    lib_pos = {s: i for i, s in enumerate(lib_smiles)}
    to_q_idx = np.full(len(exact_smiles), -1, dtype=np.int64)
    to_lib_idx = np.full(len(exact_smiles), -1, dtype=np.int64)
    for i, s in enumerate(exact_smiles):
        if s in q_pos:
            to_q_idx[i] = q_pos[s]
        if s in lib_pos:
            to_lib_idx[i] = lib_pos[s]

    idx_a, idx_b, mces = arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64), arr[:, 2]
    row, col = to_q_idx[idx_a], to_lib_idx[idx_b]
    keep = (mces != -1.0) & ~np.isnan(mces) & (row >= 0) & (col >= 0)

    gt_raw = np.full((len(q_smiles), len(lib_smiles)), np.nan, dtype=np.float32)
    gt_raw[row[keep], col[keep]] = mces[keep]
    n_resolved = int(keep.sum())
    print(f"  Raw GT matrix: {n_resolved:,} / {gt_raw.size:,} pairs resolved ({100 * n_resolved / gt_raw.size:.1f}%)")
    return gt_raw


def normalize_shard(scores_dir: Path, exact_mces_dir: Path, shard_id: int, n_shards: int, output_dir: Path,
                     metric: str = "sebastian") -> None:
    metric_fn = _NORMALIZE_METRIC_FNS[metric]
    q_smiles, lib_smiles = load_scores_smiles(scores_dir)
    gt_raw = build_raw_gt_matrix(q_smiles, lib_smiles, exact_mces_dir)

    rows, cols = np.nonzero(~np.isnan(gt_raw))
    shard_mask = (np.arange(len(rows)) % n_shards) == shard_id
    shard_rows, shard_cols = rows[shard_mask], cols[shard_mask]
    print(f"shard {shard_id}/{n_shards} ({metric}): {len(shard_rows):,} pairs to normalize")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for n, (r, c) in enumerate(zip(shard_rows, shard_cols)):
        try:
            d = metric_fn(q_smiles[int(r)], lib_smiles[int(c)])
        except Exception as e:
            # Defense-in-depth against a SIGALRM delivered a few bytecodes
            # late inside the metric function -- never let one pathological
            # pair crash the shard.
            print(f"  WARNING: pair ({r},{c}) raised {type(e).__name__}: {e} -- treating as unresolved", flush=True)
            d = None
        if d is not None:
            out.append((int(r), int(c), d))
        if (n + 1) % 2000 == 0:
            print(f"  {n + 1:,}/{len(shard_rows):,}", flush=True)

    out_arr = np.asarray(out, dtype=np.float32) if out else np.zeros((0, 3), dtype=np.float32)
    out_path = output_dir / f"shard_{shard_id:05d}.npy"
    np.save(out_path, out_arr)
    print(f"wrote {out_path} ({len(out_arr):,} resolved pairs)")


def _combine_normalize_shards(scores_dir: Path, shard_dir: Path, n_shards: int) -> np.ndarray:
    q_smiles, lib_smiles = load_scores_smiles(scores_dir)
    n_q, n_lib = len(q_smiles), len(lib_smiles)
    gt_norm = np.full((n_q, n_lib), np.nan, dtype=np.float32)

    missing = []
    total_pairs = 0
    for shard_id in range(n_shards):
        shard_path = shard_dir / f"shard_{shard_id:05d}.npy"
        if not shard_path.exists():
            missing.append(shard_id)
            continue
        arr = np.load(shard_path)
        if len(arr) == 0:
            continue
        gt_norm[arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)] = arr[:, 2]
        total_pairs += len(arr)

    if missing:
        print(f"  WARNING: {len(missing)}/{n_shards} shards missing: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    else:
        print(f"  all {n_shards} shards present")
    print(f"  combined {total_pairs:,} resolved pairs into ({n_q}, {n_lib}) matrix")
    return gt_norm


def _mask_bailout_contradictions(gt_raw: np.ndarray, gt_norm: np.ndarray) -> np.ndarray:
    """RASCAL's combinatorial MCES search can silently return a fake
    zero-overlap result for a small fraction of pairs (its own "too many
    matching bond pairs" safety cutoff, which doesn't surface as an empty
    result or a timeout flag -- see paper_metrics.py). Detected via
    cross-check against the independently-computed, bailout-immune raw GT
    MCES: a pair with a low raw MCES (near-identical molecules) but a high
    normalized MCES (apparently unrelated) is a contradiction, not a real
    result, and is set to unresolved (NaN) rather than left in as a wrong
    value."""
    contradiction = ~np.isnan(gt_norm) & (gt_raw <= BAILOUT_RAW_THRESHOLD) & (gt_norm >= BAILOUT_NORM_THRESHOLD)
    n = int(contradiction.sum())
    gt_norm = gt_norm.copy()
    gt_norm[contradiction] = np.nan
    print(f"  masked {n} bailout-contradiction pairs (raw<={BAILOUT_RAW_THRESHOLD}, norm>={BAILOUT_NORM_THRESHOLD}) to unresolved")
    return gt_norm


def normalize_combine(scores_dir: Path, exact_mces_dir: Path, shard_dir: str, n_shards: int, cache_name: str = "normalized_mces_cache.npy") -> None:
    cache_path = exact_mces_dir / cache_name
    q_smiles, lib_smiles = load_scores_smiles(scores_dir)
    gt_raw = build_raw_gt_matrix(q_smiles, lib_smiles, exact_mces_dir)

    if cache_path.exists():
        print(f"  {cache_path} already exists -- loading it instead of recombining shards")
        gt_norm = np.load(cache_path)
    else:
        gt_norm = _combine_normalize_shards(scores_dir, Path(shard_dir), n_shards)

    gt_norm = _mask_bailout_contradictions(gt_raw, gt_norm)
    np.save(cache_path, gt_norm)
    print(f"  wrote {cache_path}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build_mgf", help="CASMI + GNPS -> the two search MGFs.")
    s.add_argument("--casmi_pkl", required=True)
    s.add_argument("--training_pkl", required=True, help="edit_distance_neurips_nist_exhaustive.pkl")
    s.add_argument("--gnps_mgf", required=True)
    s.add_argument("--output_dir", required=True)

    s = sub.add_parser("exact_mces_prepare", help="Build smiles.txt/pairs.npy/meta.json for one search.")
    s.add_argument("--output_dir", type=Path, required=True)
    s.add_argument("--mgf", required=True, help="One search's MGF, FOLD=query/library")
    s.add_argument("--n_blocks", type=int, default=100)

    s = sub.add_parser("exact_mces_compute_block", help="Compute one block of exact raw MCES.")
    s.add_argument("--output_dir", type=Path, required=True)
    s.add_argument("--task_id", type=int, required=True)
    s.add_argument("--n_jobs", type=int, default=-1, help="-1 = SLURM_CPUS_PER_TASK or CPU affinity count, never the whole host.")
    s.add_argument("--timeout", type=float, default=None, help="Per-pair wall-clock deadline (s).")

    s = sub.add_parser("exact_mces_status", help="Print exact-MCES block completion progress.")
    s.add_argument("--output_dir", type=Path, required=True)

    s = sub.add_parser("exact_mces_combine", help="Merge exact-MCES blocks into mces_exact.npy.")
    s.add_argument("--output_dir", type=Path, required=True)

    s = sub.add_parser("normalize_shard", help="Compute one shard of normalized (RASCAL) MCES.")
    s.add_argument("--scores_dir", type=Path, required=True)
    s.add_argument("--exact_mces_dir", type=Path, required=True)
    s.add_argument("--shard_id", type=int, required=True)
    s.add_argument("--n_shards", type=int, required=True)
    s.add_argument("--output_dir", type=Path, required=True, help="Where this shard's .npy is written.")
    s.add_argument("--metric", choices=["sebastian", "johnson"], default="sebastian",
                   help="sebastian = paper's actual tier-similarity formula; johnson = paper Methods text's stated formula")

    s = sub.add_parser("normalize_combine", help="Combine normalize_shard outputs + mask bailout artifacts.")
    s.add_argument("--scores_dir", type=Path, required=True)
    s.add_argument("--exact_mces_dir", type=Path, required=True)
    s.add_argument("--shard_dir", default=None, help="only needed if the cache doesn't exist yet")
    s.add_argument("--n_shards", type=int, default=2000)
    s.add_argument("--cache_name", default="normalized_mces_cache.npy", help="cache filename within exact_mces_dir (use a different name per metric)")

    a = p.parse_args()
    if a.cmd == "build_mgf":
        build_mgf(a.casmi_pkl, a.training_pkl, a.gnps_mgf, a.output_dir)
    elif a.cmd == "exact_mces_prepare":
        exact_mces_prepare(a.output_dir, a.mgf, a.n_blocks)
    elif a.cmd == "exact_mces_compute_block":
        exact_mces_compute_block(a.output_dir, a.task_id, a.n_jobs, a.timeout)
    elif a.cmd == "exact_mces_status":
        exact_mces_status(a.output_dir)
    elif a.cmd == "exact_mces_combine":
        exact_mces_combine(a.output_dir)
    elif a.cmd == "normalize_shard":
        normalize_shard(a.scores_dir, a.exact_mces_dir, a.shard_id, a.n_shards, a.output_dir, a.metric)
    elif a.cmd == "normalize_combine":
        normalize_combine(a.scores_dir, a.exact_mces_dir, a.shard_dir, a.n_shards, a.cache_name)


if __name__ == "__main__":
    main()
