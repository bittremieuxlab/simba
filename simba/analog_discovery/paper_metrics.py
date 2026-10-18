"""Two normalized-MCES metrics, computed side by side for comparison:

- normalized_mces_distance_johnson: the paper Methods text's own stated
  Johnson-similarity formula, d = 1 - (E_c+V_c)^2/((E1+V1)(E2+V2)), with
  E_c/V_c taken from RASCAL's actual found MCES (bondMatches/atomMatches).
- normalized_mces_distance_sebastian: what his own notebook
  (bittremieuxlab/simba, simba_v1, notebooks/analog_discovery_casmi_msnlib.ipynb)
  actually computes: 1 - calculate_mces_sim(...), where calculate_mces_sim
  returns RASCAL's tier2Sim (falling back to tier1Sim).

These are NOT the same metric, despite both being called "normalized MCES
distance" in different places. tier1Sim/tier2Sim are RASCAL's own fast
pre-screening UPPER-BOUND ESTIMATES of the true Johnson similarity (see
RDKit's own RASCAL blog post, and confirmed directly from RDKit's C++
source: both use the identical (v+e)^2/denom formula shape, but with v/e
estimated from degree-sequence/bond-label heuristics rather than the true
matched subgraph) -- meant only to decide whether the expensive exact
search is worth running at all, not as a final reported value. Since
they're upper bounds on similarity, 1-tier_sim systematically
UNDERESTIMATES the true distance -- confirmed empirically: his metric
runs at roughly half our Johnson formula's value on the same close pairs,
and is naturally immune to the "too many matching bond pairs" bailout that
the Johnson formula needs masking/guarding for (tier1/tier2 are computed
before the combinatorial search that bailout afflicts).
"""

import resource
import signal

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdRascalMCES

# Even with ringMatchesRingOnly=True (below), a small fraction of pairs are
# still pathological: confirmed empirically that one pair drove a single
# process to 40GB+ RSS and hung for minutes with RASCAL's own `timeout`
# option (60s) not actually bounding it. RLIMIT_AS caps the process's
# virtual memory so such a call fails fast with a clean MemoryError instead
# of ballooning (observed: ~1.7s to fail at a 3GB cap, vs. still running
# after 3+ minutes unbounded) -- this is the primary defense. SIGALRM is a
# backup for any hang that doesn't go through a large allocation. Both are
# set once per process (the limit/handler persist for the process's
# lifetime -- cheap to set repeatedly, but no need to).
_HARD_MEM_LIMIT_BYTES = 3 * 1024**3
_HARD_TIMEOUT_SECONDS = 5
_limits_installed = False


class _MCESHardTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _MCESHardTimeout()


def _install_hard_limits():
    global _limits_installed
    if _limits_installed:
        return
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_HARD_MEM_LIMIT_BYTES, _HARD_MEM_LIMIT_BYTES))
    except (ValueError, resource.error):
        pass  # best-effort; some platforms/containers disallow RLIMIT_AS
    signal.signal(signal.SIGALRM, _alarm_handler)
    _limits_installed = True


def normalized_mces_distance_johnson(smiles1: str, smiles2: str) -> float | None:
    """Paper Methods text's own Johnson-similarity formula, computed from
    RASCAL's actual found MCES (bondMatches/atomMatches).

    Returns None if either SMILES fails to parse, the MCES search fails,
    or the pair hits the hard memory/time limit (pathological RASCAL case).
    """
    _install_hard_limits()
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None

    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0.0  # always return a MCES, even for very dissimilar pairs
    opts.returnEmptyMCES = True
    # Without this, RASCAL's combinatorial bond-pair search explodes for
    # larger/symmetric molecules and silently bails out ("Too many matching
    # bond pairs (N) so can't continue"), returning a non-empty result with
    # zero matches -- indistinguishable from a genuine no-overlap pair.
    # Restricting ring bonds to only match ring bonds collapses the search
    # space and avoids the bailout.
    opts.ringMatchesRingOnly = True
    opts.timeout = 2
    signal.alarm(_HARD_TIMEOUT_SECONDS)
    try:
        results = rdRascalMCES.FindMCES(mol1, mol2, opts)
        if not results:
            return None
        r = results[0]
        if r.timedOut:
            # A timed-out search can still return a non-empty result with
            # few/zero matches, silently understating the true overlap.
            return None
        e_c, v_c = len(r.bondMatches()), len(r.atomMatches())
    except Exception:
        return None
    finally:
        signal.alarm(0)

    e1, v1 = mol1.GetNumBonds(), mol1.GetNumAtoms()
    e2, v2 = mol2.GetNumBonds(), mol2.GetNumAtoms()
    denom = (e1 + v1) * (e2 + v2)
    if denom == 0:
        return None
    return 1.0 - ((e_c + v_c) ** 2) / denom


def normalized_mces_distance_sebastian(smiles1: str, smiles2: str) -> float | None:
    """What Sebastian's own notebook actually computes: 1 - calculate_mces_sim(...),
    i.e. 1 - (tier2Sim, falling back to tier1Sim) -- RASCAL's fast
    pre-screening upper-bound similarity estimate, not the true Johnson
    similarity from an actual found MCES.

    Returns None if either SMILES fails to parse, the MCES search fails,
    or the pair hits the hard memory/time limit (pathological RASCAL case,
    guarded defensively even though tier-similarity is empirically immune
    to the bondMatches-based bailout the Johnson formula needs masking for,
    since tier1/tier2 are computed before the combinatorial search begins).
    """
    _install_hard_limits()
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None

    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0.7  # matches calculate_mces_sim's default
    opts.returnEmptyMCES = True
    # Sebastian's calculate_mces_sim doesn't set ringMatchesRingOnly or a
    # tightened timeout. We keep our own timeout tightened (his default is
    # RASCAL's own 60s) purely for throughput at scale -- doesn't change
    # which result is returned once it completes, same as before.
    opts.timeout = 2
    signal.alarm(_HARD_TIMEOUT_SECONDS)
    try:
        results = rdRascalMCES.FindMCES(mol1, mol2, opts)
        if not results:
            return None
        r = results[0]
        tier1, tier2 = r.tier1Sim, r.tier2Sim
        sim = tier2 if tier2 != -1 else tier1
    except Exception:
        return None
    finally:
        signal.alarm(0)

    return 1.0 - sim


def rank_by_mces_then_edit_distance(
    pred_mces: np.ndarray, pred_edit_distance_class: np.ndarray
) -> np.ndarray:
    """Candidate ranking matching Sebastian's own compute_ranking: primarily
    by predicted MCES ROUNDED TO THE NEAREST INTEGER (not the continuous
    value -- this matters, since it creates many more ties than continuous
    floats ever would, making the next tiebreak do real work rather than
    being a rare edge case), ties broken by predicted edit-distance class,
    remaining ties broken by the continuous MCES value.

    Edit-distance class direction: confirmed via his own sanity-check code
    ("Edit distance found: {5 - argmax(class)} modifications") that class 5
    = 0 modifications (most similar) and class 0 = 5 modifications (least
    similar) -- i.e. HIGHER class = closer analog. This is the opposite of
    what an earlier version of this function assumed (lower = closer).

    Parameters
    ----------
    pred_mces : (n_query, n_library) float array, raw MCES prediction
        (lower = closer analog).
    pred_edit_distance_class : (n_query, n_library) int array, predicted
        ordinal edit-distance class 0..5 (HIGHER = closer analog).

    Returns
    -------
    (n_query, n_library) int array: for each query row, the rank of each
    library candidate (0 = best/closest).
    """
    pred_mces = np.asarray(pred_mces)
    pred_edit_distance_class = np.asarray(pred_edit_distance_class)
    assert pred_mces.shape == pred_edit_distance_class.shape

    pred_mces_int = np.round(pred_mces)

    n_q, n_lib = pred_mces.shape
    ranks = np.empty((n_q, n_lib), dtype=np.int32)
    for i in range(n_q):
        # np.lexsort: last key is primary. Primary: integer MCES ascending.
        # Secondary: edit-distance class descending (higher = closer), via
        # negation. Tertiary: continuous MCES ascending.
        order = np.lexsort((pred_mces[i], -pred_edit_distance_class[i], pred_mces_int[i]))
        ranks[i, order] = np.arange(n_lib)
    return ranks
