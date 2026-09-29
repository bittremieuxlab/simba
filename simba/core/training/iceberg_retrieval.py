"""SIMBA + ICEBERG retrieval Hit@k: candidate data loading, embedding, and
ranking logic shared by IcebergHitRateCallback (periodic, live-training-
time check) and tools/compute_iceberg_cosine_baseline.py (the model-free
cosine baseline for the same benchmark).

For each real test spectrum: rank its formula-matched candidates (each
represented by an ICEBERG-predicted in-silico spectrum, since most
candidates have no real measured spectrum) by SIMBA embedding similarity
-- raw cosine, and (when the model has an mces_bucket head) CORN-corrected
MCES -- and check whether the true molecule lands in the top 1/5/20.
"""

import copy
import json

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from tqdm import tqdm

from simba.core.chemistry.chem_utils import ADDUCT_TO_MASS, theoretical_precursor_mz
from simba.core.data.encoding import encode_adduct_mass
from simba.core.data.preprocessor import Preprocessor
from simba.core.data.spectrum import SpectrumExt


MAX_PEAKS = 100  # must match model.transformer.context_length
N_ADDUCTS = len(ADDUCT_TO_MASS)

DEFAULT_MGF = "/sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test.mgf"
DEFAULT_CANDIDATES = (
    "/sofia/projects/2026_053/spectrawl_project/data/massspecgym/"
    "MassSpecGym_retrieval_candidates_formula.json"
)
DEFAULT_CANDIDATE_TSV = [
    "/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_existing_overlap.tsv",
    "/sofia/projects/2026_053/simba_project/ICEBERG/data/candidates_gaetan_test_new.tsv",
]
DEFAULT_ICEBERG_PREDS = [
    "/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_test_official/preds.hdf5",
    "/sofia/projects/2026_053/simba_project/ICEBERG/results/candidates_gaetan_test_new/preds.hdf5",
]


# ── Spectrum loading ─────────────────────────────────────────────────────────


def canonicalize(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def _to_spectrum_ext(spec, fold: str, precursor_mass_mode: str = "theoretical"):
    charge = spec.get("charge")
    charge = int(charge) if charge else 1
    ce_val = spec.get("ce") or spec.get("collision_energy")
    ce = None
    if ce_val is not None:
        try:
            ce = int(float(ce_val))
        except (ValueError, TypeError):
            ce = None
    ionmode = spec.get("ionmode")
    ionmode = ionmode.lower() if ionmode else "none"

    if precursor_mass_mode == "theoretical":
        mol = Chem.MolFromSmiles(spec.get("smiles"))
        precursor_mz = theoretical_precursor_mz(ExactMolWt(mol), spec.get("adduct"))
    else:
        precursor_mz = float(spec.get("precursor_mz") or 0.0)

    return SpectrumExt(
        identifier=str(spec.get("scans") or ""),
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        mz=np.asarray(spec.peaks.mz, dtype=np.float64),
        intensity=np.asarray(spec.peaks.intensities, dtype=np.float64),
        retention_time=np.nan,
        params={},
        library=None,
        inchi=None,
        smiles=spec.get("smiles"),
        ionmode=ionmode,
        adduct=spec.get("adduct"),
        ce=ce,
        ion_activation=None,
        ionization_method=None,
        bms=None,
        superclass=None,
        classe=None,
        subclass=None,
        fold=fold,
    )


def load_spectra(mgf_path: str, fold: str, precursor_mass_mode: str = "theoretical"):
    import matchms

    smiles_list, spectra = [], []
    for spec in tqdm(
        matchms.importing.load_from_mgf(mgf_path), desc=f"Loading {fold}", unit="spec"
    ):
        if spec.get("fold") != fold:
            continue
        smi = spec.get("smiles")
        if not smi or spec.peaks is None or len(spec.peaks.mz) < 1:
            continue
        smiles_list.append(smi)
        spectra.append(_to_spectrum_ext(spec, fold, precursor_mass_mode))
    return smiles_list, spectra


def _encode_ionmode(ionmode_val) -> float:
    if ionmode_val is None or ionmode_val == "None":
        return 0.0
    return 1.0 if ionmode_val == "positive" else -1.0


def spectra_to_tensors(specs: list, device: torch.device):
    pp = Preprocessor()
    b = len(specs)
    mz_t = torch.zeros(b, MAX_PEAKS, dtype=torch.float32)
    int_t = torch.zeros(b, MAX_PEAKS, dtype=torch.float32)
    prec_t = torch.zeros(b, dtype=torch.float32)
    charge_t = torch.ones(b, dtype=torch.float32)
    ionmode_t = torch.zeros(b, dtype=torch.float32)
    adduct_t = torch.zeros(b, N_ADDUCTS, dtype=torch.float32)
    ce_t = torch.zeros(b, dtype=torch.float32)

    for i, spec in enumerate(specs):
        processed = pp.preprocess_spectrum(
            copy.copy(spec),
            fragment_tol_mass=10,
            fragment_tol_mode="ppm",
            min_intensity=0.01,
            max_num_peaks=MAX_PEAKS,
            scale_intensity=None,
        )
        mz = np.asarray(processed.mz, dtype=np.float32)
        intensity = np.asarray(processed.intensity, dtype=np.float32)
        n = len(mz)
        mz_t[i, :n] = torch.from_numpy(mz)
        int_t[i, :n] = torch.from_numpy(intensity)
        prec_t[i] = float(spec.precursor_mz or 0.0)
        if spec.precursor_charge:
            charge_t[i] = float(spec.precursor_charge)
        ionmode_t[i] = _encode_ionmode(spec.ionmode)
        if spec.adduct:
            adduct_t[i] = torch.tensor(
                encode_adduct_mass(spec.adduct), dtype=torch.float32
            )
        ce_t[i] = 0.0 if spec.ce is None else float(spec.ce)

    int_t = torch.sqrt(int_t.clamp(min=0.0))
    norms = int_t.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
    int_t = int_t / norms
    meta = {
        "ionmode": ionmode_t.to(device),
        "adduct": adduct_t.to(device),
        "ce": ce_t.to(device),
    }
    return (
        mz_t.to(device),
        int_t.to(device),
        prec_t.to(device),
        charge_t.to(device),
        meta,
    )


@torch.no_grad()
def embed_spectra(model, spectra: list, batch_size: int, device: torch.device):
    """Encode spectra through SIMBA's encoder. cosine_no_head means the
    similarity head is the identity, so the relu'd CLS-token embedding IS
    both the raw embedding (compute_from_embeddings' emb0/emb1) and, once
    L2-normalized, the cosine-similarity embedding. Returns
    (normalized, raw, tokens, valid_mask) -- tokens/valid_mask are only
    populated (else None) when model.use_filip_head is set: the per-peak
    embeddings (relu'd, CLS excluded) and their validity mask (True = real
    peak, False = padding), stored on CPU in fp16 to keep the full
    candidate-pool footprint manageable (~600k candidates x 100 peaks x
    d_model is tens of GB even at fp16 -- infeasible to keep resident on
    GPU, fine on CPU RAM)."""
    enc = model.spectrum_encoder
    use_ion_mode = getattr(enc, "use_ion_mode", False)
    use_adduct = getattr(enc, "use_adduct", False)
    use_ce = getattr(enc, "use_ce", False)
    use_filip = getattr(model, "use_filip_head", False)

    all_embs, all_raw = [], []
    all_tokens, all_valid = [], []
    for start in tqdm(
        range(0, len(spectra), batch_size), desc="Embedding", unit="batch"
    ):
        batch = spectra[start : start + batch_size]
        mz, intensity, prec, charge, meta = spectra_to_tensors(batch, device)
        kwargs = {"precursor_mass": prec, "precursor_charge": charge}
        if use_ion_mode:
            kwargs["ionmode"] = meta["ionmode"]
        if use_adduct:
            kwargs["adduct"] = meta["adduct"]
        if use_ce:
            kwargs["ce"] = meta["ce"]
        latent, pad_mask = model.spectrum_encoder(
            mz_array=mz, intensity_array=intensity, **kwargs
        )
        emb = model.relu(latent[:, 0, :])
        all_raw.append(emb.cpu())
        all_embs.append(F.normalize(emb, p=2, dim=-1).cpu())
        if use_filip:
            tokens = model.relu(latent[:, 1:, :])
            valid = ~pad_mask[:, 1:]
            all_tokens.append(tokens.half().cpu())
            all_valid.append(valid.cpu())

    tokens_out = torch.cat(all_tokens) if all_tokens else None
    valid_out = torch.cat(all_valid) if all_valid else None
    return torch.cat(all_embs), torch.cat(all_raw), tokens_out, valid_out


# ── Candidates ───────────────────────────────────────────────────────────────


def load_candidate_index(candidate_tsv) -> pd.DataFrame:
    paths = [candidate_tsv] if isinstance(candidate_tsv, str) else list(candidate_tsv)
    dfs = [pd.read_csv(p, sep="\t") for p in paths]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    return df.set_index(["smiles", "ionization"])


def load_iceberg_spectra(iceberg_preds, cand_ids: list) -> dict:
    paths = [iceberg_preds] if isinstance(iceberg_preds, str) else list(iceberg_preds)
    result: dict = {}
    wanted = set(cand_ids)
    for path in paths:
        with h5py.File(path, "r") as f:
            manifest = f["__predspec_manifest__"]
            name_to_leaf = {}
            for name, leaf in zip(manifest["name"][:], manifest["leaf_path"][:]):
                name_to_leaf[name.decode().removeprefix("pred_")] = leaf.decode()
            for name, leaf in tqdm(
                name_to_leaf.items(), desc=f"Loading ICEBERG spectra ({path})"
            ):
                if name not in wanted:
                    continue
                arr = f[leaf]["f"][:]
                mask = arr[:, 0] > 0
                result[name] = (
                    arr[mask, 0].astype(np.float64),
                    arr[mask, 1].astype(np.float64),
                )
    return result


def build_candidate_spectra(cand_index: pd.DataFrame, iceberg_specs: dict):
    smiles_out, specs_out = [], []
    for (smi, adduct), row in cand_index.iterrows():
        cand_id = row["spec"]
        if cand_id not in iceberg_specs:
            continue
        mz, intensity = iceberg_specs[cand_id]
        specs_out.append(
            SpectrumExt(
                identifier=cand_id,
                precursor_mz=float(row["precursor"]),
                precursor_charge=1,
                mz=mz,
                intensity=intensity,
                retention_time=np.nan,
                params={},
                library=None,
                inchi=None,
                smiles=smi,
                ionmode="positive",
                adduct=adduct,
                ce=35,
                ion_activation=None,
                ionization_method=None,
                bms=None,
                superclass=None,
                classe=None,
                subclass=None,
                fold="candidate_pred",
            )
        )
        smiles_out.append(smi)
    return smiles_out, specs_out


def load_candidates(candidates_json: str) -> dict:
    with open(candidates_json) as fh:
        candidate_json = json.load(fh)
    return {canonicalize(k): v for k, v in candidate_json.items()}


def load_all_iceberg_data(mgf, candidates_json, candidate_tsv, iceberg_preds):
    """One-time load of everything needed for this benchmark: real test
    spectra, the candidate pool per query, and every candidate's
    ICEBERG-predicted spectrum. None of this depends on model weights, so
    it's safe to load once and reuse across many checkpoints/epochs."""
    test_smiles, test_spectra = load_spectra(mgf, "test")
    test_adducts = [s.adduct for s in test_spectra]

    query_candidates = load_candidates(candidates_json)

    cand_index = load_candidate_index(candidate_tsv)
    all_cand_ids = cand_index["spec"].tolist()
    iceberg_specs = load_iceberg_spectra(iceberg_preds, all_cand_ids)
    cand_smiles, cand_spectra = build_candidate_spectra(cand_index, iceberg_specs)
    cand_smi_to_row = {
        (smi, spec.adduct): i
        for i, (smi, spec) in enumerate(zip(cand_smiles, cand_spectra))
    }
    return {
        "test_smiles": test_smiles,
        "test_spectra": test_spectra,
        "test_adducts": test_adducts,
        "query_candidates": query_candidates,
        "cand_spectra": cand_spectra,
        "cand_smi_to_row": cand_smi_to_row,
    }


# ── Ranking + Hit@k ──────────────────────────────────────────────────────────


def rank_candidates_filip(
    test_smiles,
    test_adducts,
    query_candidates,
    cand_smi_to_row,
    test_tokens,
    test_valid,
    cand_tokens,
    cand_valid,
    model,
    device,
    top_k=20,
):
    """Same ranking as rank_candidates, but scores each (query, candidate)
    pair with the model's own FILIP fine-grained token similarity instead
    of a plain dot product of pooled embeddings -- reuses
    model._filip_similarity directly, so this is exactly the same scoring
    mechanism used during training/validation. Only the query's own
    (typically small) candidate subset is ever moved to GPU at once, since
    the full candidate-pool token tensors live on CPU."""
    per_query = []
    for i, (q_smi, q_adduct) in enumerate(zip(test_smiles, test_adducts)):
        cand_list = query_candidates.get(canonicalize(q_smi), [])
        row_idxs, cand_smis = [], []
        for c in cand_list:
            row_idx = cand_smi_to_row.get((c, q_adduct))
            if row_idx is None:
                continue
            row_idxs.append(row_idx)
            cand_smis.append(c)
        if not row_idxs:
            per_query.append(None)
            continue

        n = len(row_idxs)
        q_tok = test_tokens[i : i + 1].expand(n, -1, -1).to(device).float()
        q_valid = test_valid[i : i + 1].expand(n, -1).to(device)
        c_tok = cand_tokens[row_idxs].to(device).float()
        c_valid = cand_valid[row_idxs].to(device)
        with torch.no_grad():
            sims = model._filip_similarity(q_tok, q_valid, c_tok, c_valid)
        order = torch.argsort(sims, descending=True)
        per_query.append([cand_smis[j] for j in order.tolist()][:top_k])
    return per_query


def rank_candidates(
    test_smiles,
    test_adducts,
    query_candidates,
    cand_smi_to_row,
    test_embs,
    cand_embs,
    top_k=20,
):
    per_query = []
    for i, (q_smi, q_adduct) in enumerate(zip(test_smiles, test_adducts)):
        cand_list = query_candidates.get(canonicalize(q_smi), [])
        row_idxs, cand_smis = [], []
        for c in cand_list:
            row_idx = cand_smi_to_row.get((c, q_adduct))
            if row_idx is None:
                continue
            row_idxs.append(row_idx)
            cand_smis.append(c)
        if not row_idxs:
            per_query.append(None)
            continue
        sims = (test_embs[i : i + 1] @ cand_embs[row_idxs].T).squeeze(0)
        order = torch.argsort(sims, descending=True)
        per_query.append([cand_smis[j] for j in order.tolist()][:top_k])
    return per_query


def rank_candidates_corn_corrected(
    test_smiles,
    test_adducts,
    query_candidates,
    cand_smi_to_row,
    test_embs_raw,
    cand_embs_raw,
    model,
    device,
    top_k=20,
    test_tokens=None,
    test_valid=None,
    cand_tokens=None,
    cand_valid=None,
):
    """Ranks by CORN-corrected MCES. When the model has a FILIP head,
    emb_sim_2 (the primary MCES score) must be computed from the model's
    own fine-grained token similarity, not left as a plain dot product of
    pooled embeddings -- otherwise compute_from_embeddings would silently
    return emb_sim_2=None for any FILIP model (it only uses `filip_score`
    when use_filip_head is set, and this function used to never pass one).
    Likewise, when mces_bucket_use_filip_tokens is set, emb_sim_3 (the
    bucket logits) must be built from the same token-mean-pooled
    representation used during training, not the CLS embeddings -- so
    test_tokens/test_valid/cand_tokens/cand_valid (from embed_spectra) are
    threaded through here exactly as rank_candidates_filip already does for
    the primary score."""
    from simba.core.training.callbacks import (
        _corn_corrected_mces,
        _corn_corrected_ranking_score,
    )

    use_filip = getattr(model, "use_filip_head", False)
    bin_edges = model.mces_bucket_bin_edges.cpu().numpy()
    per_query = []
    for i, (q_smi, q_adduct) in enumerate(zip(test_smiles, test_adducts)):
        cand_list = query_candidates.get(canonicalize(q_smi), [])
        row_idxs, cand_smis = [], []
        for c in cand_list:
            row_idx = cand_smi_to_row.get((c, q_adduct))
            if row_idx is None:
                continue
            row_idxs.append(row_idx)
            cand_smis.append(c)
        if not row_idxs:
            per_query.append(None)
            continue

        n = len(row_idxs)
        emb0 = test_embs_raw[i : i + 1].expand(n, -1).to(device)
        emb1 = cand_embs_raw[row_idxs].to(device)

        tok0 = tok1 = vl0 = vl1 = None
        filip_score = None
        if use_filip:
            tok0 = test_tokens[i : i + 1].expand(n, -1, -1).to(device).float()
            vl0 = test_valid[i : i + 1].expand(n, -1).to(device)
            tok1 = cand_tokens[row_idxs].to(device).float()
            vl1 = cand_valid[row_idxs].to(device)
            with torch.no_grad():
                filip_score = model._filip_similarity(tok0, vl0, tok1, vl1)

        with torch.no_grad():
            emb_sim_2, emb_sim_3 = model.compute_from_embeddings(
                emb0,
                emb1,
                filip_score=filip_score,
                tokens0=tok0,
                valid0=vl0,
                tokens1=tok1,
                valid1=vl1,
            )
        pred_mces = ((1.0 - emb_sim_2) * model.mces_max_value).cpu().numpy()
        bucket_pred = model._corn_decode_bin_generic(emb_sim_3).cpu().numpy()
        corrected = _corn_corrected_mces(pred_mces, bucket_pred, bin_edges)
        score = _corn_corrected_ranking_score(corrected, pred_mces)

        order = np.argsort(score)  # ascending: lowest (closest) MCES first
        per_query.append([cand_smis[j] for j in order][:top_k])
    return per_query


def compute_hit_rates_from_ranking(test_smiles, per_query_ranked, ks=(1, 5, 20)):
    hits = dict.fromkeys(ks, 0)
    n = 0
    n_no_candidates = 0
    for q_smi, ranked in zip(test_smiles, per_query_ranked):
        if ranked is None:
            n_no_candidates += 1
            continue
        q_canon = canonicalize(q_smi)
        for k in ks:
            if any(canonicalize(c) == q_canon for c in ranked[:k]):
                hits[k] += 1
        n += 1
    return {k: (hits[k] / n if n > 0 else 0.0) for k in ks}, n_no_candidates


def compute_iceberg_hit_rates(
    model, device, data: dict, batch_size: int = 512, ks=(1, 5, 20)
):
    """(raw_hits, corrected_hits): embeds test + candidate spectra with the
    given model's current weights and computes Hit@k both ways.
    corrected_hits is None when the model has no mces_bucket head. When the
    model has a FILIP head, raw_hits is ranked with the model's own
    fine-grained token similarity instead of pooled-embedding cosine."""
    test_embs, test_embs_raw, test_tokens, test_valid = embed_spectra(
        model, data["test_spectra"], batch_size, device
    )
    cand_embs, cand_embs_raw, cand_tokens, cand_valid = embed_spectra(
        model, data["cand_spectra"], batch_size, device
    )

    if getattr(model, "use_filip_head", False):
        ranked_raw = rank_candidates_filip(
            data["test_smiles"],
            data["test_adducts"],
            data["query_candidates"],
            data["cand_smi_to_row"],
            test_tokens,
            test_valid,
            cand_tokens,
            cand_valid,
            model,
            device,
        )
    else:
        ranked_raw = rank_candidates(
            data["test_smiles"],
            data["test_adducts"],
            data["query_candidates"],
            data["cand_smi_to_row"],
            test_embs,
            cand_embs,
        )
    raw_hits, _ = compute_hit_rates_from_ranking(data["test_smiles"], ranked_raw, ks)

    corrected_hits = None
    if model.use_mces_bucket_head:
        ranked_corrected = rank_candidates_corn_corrected(
            data["test_smiles"],
            data["test_adducts"],
            data["query_candidates"],
            data["cand_smi_to_row"],
            test_embs_raw,
            cand_embs_raw,
            model,
            device,
            test_tokens=test_tokens,
            test_valid=test_valid,
            cand_tokens=cand_tokens,
            cand_valid=cand_valid,
        )
        corrected_hits, _ = compute_hit_rates_from_ranking(
            data["test_smiles"], ranked_corrected, ks
        )
    return raw_hits, corrected_hits
