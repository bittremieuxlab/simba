#!/usr/bin/env python
"""Full query x library-molecule ground-truth matrix with Sebastian's exact MCES
similarity (calculate_mces_sim_exact), for the analog-discovery plots.

Standalone: needs only python>=3.9, numpy and rdkit (use rdkit==2023.9.4,
the version of the sofia training venv). No simba import.

Metric per pair (his snippet, plus two agreed deviations):
  RASCAL FindMCES, similarityThreshold=0.0, returnEmptyMCES=True, timeout=10 s,
  ringMatchesRingOnly left at its default (False) -> result.similarity
  (Johnson similarity of the actual MCES; distance = 1 - similarity).
  Deviation 1: if the 10 s search times out and tier1Sim >= 0.7 (a possible
    close analog), retry once with timeout=60 s; still timed out -> NaN.
    Timeouts with tier1Sim < 0.7 -> NaN (as in his snippet: None).
  Deviation 2: RASCAL's "Too many matching bond pairs ... can't continue"
    bailout returns 0 matched bonds WITHOUT setting timedOut; his snippet would
    return similarity 0. Detected from RASCAL's stderr and stored as NaN.

Work unit = block of BLOCK_SIZE consecutive library molecules for one query.
Blocks are written atomically and skipped when present, so any task can be
re-run/resubmitted safely. Task t of N processes blocks with index % N == t,
in parallel over --workers processes.

Subcommands:
  selftest  check RDKit version + bailout detection on two known pairs
  run       compute blocks (one SLURM array task)
  rerun     recompute pairs with given statuses (default hard_fail) in finished
            blocks, e.g. with a larger --mem_per_worker_gb and fewer --workers
  status    count done/missing blocks
  combine   assemble per-search matrices + summary (after all blocks are done)

Status codes (gt_status.npy):
  0 ok | 1 smiles parse failure | 2 no RASCAL result | 3 timeout 10 s, tier1<0.7
  4 timeout after 60 s retry | 5 bailout | 6 hard failure (watchdog/crash/memory)
  7 ok, needed the 60 s retry
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import resource
import signal
import socket
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SEARCHES = ("A", "B")
BLOCK_SIZE = 2000
TIMEOUT_1, TIMEOUT_RETRY, RETRY_TIER1 = 10, 60, 0.7
HARD_TIMEOUT = 150  # watchdog per pair (s); > TIMEOUT_1 + TIMEOUT_RETRY + overhead
OK, PARSE, NO_RESULT, TIMEOUT, TIMEOUT_RETRY_FAIL, BAILOUT, HARD, OK_RETRY = range(8)
STATUS_NAMES = ["ok", "parse_fail", "no_result", "timeout_10s", "timeout_after_retry", "bailout", "hard_fail", "ok_after_retry"]
BAILOUT_MSG = b"Too many matching bond pairs"


# ---------------------------------------------------------------- worker side
def _init_worker(mem_gb):
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if mem_gb:
        limit = int(mem_gb * 1024**3)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ValueError, OSError):
            pass
    # RASCAL prints its bailout message straight to C++ std::cerr: give each
    # worker its own stderr file and look for the message after every call.
    tmp = tempfile.TemporaryFile()
    os.dup2(tmp.fileno(), 2)


def _find_mces(mol1, mol2, timeout):
    from rdkit.Chem import rdRascalMCES

    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0.0
    opts.returnEmptyMCES = True
    opts.timeout = timeout
    start = os.fstat(2).st_size
    results = rdRascalMCES.FindMCES(mol1, mol2, opts)
    end = os.fstat(2).st_size
    bailed = end > start and BAILOUT_MSG in os.pread(2, end - start, start)
    return results, bailed


def mces_exact(pair):
    """-> (similarity, tier1Sim, tier2Sim, status)."""
    from rdkit import Chem

    nan = float("nan")
    try:
        mol1, mol2 = Chem.MolFromSmiles(pair[0]), Chem.MolFromSmiles(pair[1])
        if mol1 is None or mol2 is None:
            return nan, nan, nan, PARSE
        results, bailed = _find_mces(mol1, mol2, TIMEOUT_1)
        if not results:
            return nan, nan, nan, NO_RESULT
        r = results[0]
        tier1, tier2, status = r.tier1Sim, r.tier2Sim, OK
        if r.timedOut:
            if tier1 < RETRY_TIER1:
                return nan, tier1, tier2, TIMEOUT
            results, bailed = _find_mces(mol1, mol2, TIMEOUT_RETRY)
            if not results:
                return nan, tier1, tier2, NO_RESULT
            r = results[0]
            if r.timedOut:
                return nan, tier1, tier2, TIMEOUT_RETRY_FAIL
            status = OK_RETRY
        if bailed:
            return nan, tier1, tier2, BAILOUT
        return r.similarity, tier1, tier2, status
    except Exception:  # MemoryError from RLIMIT_AS, RDKit errors
        return nan, nan, nan, HARD


def mces_exact_timed(pair):
    t0 = time.time()
    return (*mces_exact(pair), time.time() - t0)


# ---------------------------------------------------------------- driver side
def read_lines(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def load_inputs(inputs_dir, searches):
    data = {}
    for s in searches:
        d = Path(inputs_dir) / f"search_{s}"
        data[s] = (read_lines(d / "queries.txt"), read_lines(d / "molecules.txt"))
    return data


def enumerate_blocks(data):
    """Deterministic global block list: (search, query_idx, mol_start, mol_end)."""
    blocks = []
    for s, (queries, molecules) in data.items():
        n_chunks = math.ceil(len(molecules) / BLOCK_SIZE)
        for qi in range(len(queries)):
            for c in range(n_chunks):
                blocks.append((s, qi, c * BLOCK_SIZE, min((c + 1) * BLOCK_SIZE, len(molecules))))
    return blocks


def block_path(out_dir, block):
    s, qi, start, _ = block
    return Path(out_dir) / f"search_{s}" / "blocks" / f"q{qi:04d}_m{start:06d}.npz"


class PoolHolder:
    def __init__(self, workers, mem_gb):
        self.workers, self.mem_gb, self.pool = workers, mem_gb, None

    def get(self):
        if self.pool is None:
            self.pool = mp.Pool(self.workers, initializer=_init_worker, initargs=(self.mem_gb,), maxtasksperchild=5000)
        return self.pool

    def reset(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
        self.pool = None


def stream_pairs(holder, pairs, on_result):
    """Run all pairs through one continuous imap stream (no idle cores between
    blocks); on_result(k, (sim, tier1, tier2, status, seconds)) in input order.
    A pair that hangs or kills its worker is recorded as HARD after
    HARD_TIMEOUT s and the pool is restarted right after it."""
    k = 0
    while k < len(pairs):
        results = holder.get().imap(mces_exact_timed, pairs[k:], chunksize=1)
        try:
            for k in range(k, len(pairs)):
                on_result(k, results.next(timeout=HARD_TIMEOUT))
            k = len(pairs)
        except mp.TimeoutError:
            on_result(k, (np.nan, np.nan, np.nan, HARD, float(HARD_TIMEOUT)))
            k += 1
            holder.reset()


def new_block_arrays(n):
    return {"sim": np.full(n, np.nan, np.float32), "tier1": np.full(n, np.nan, np.float32),
            "tier2": np.full(n, np.nan, np.float32), "status": np.zeros(n, np.uint8),
            "seconds": np.zeros(n, np.float32)}


def save_block(path, arrays, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays, meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def cmd_run(args):
    import rdkit

    data = load_inputs(args.inputs_dir, args.searches)
    blocks = enumerate_blocks(data)
    mine = [b for idx, b in enumerate(blocks) if idx % args.n_tasks == args.task_id]
    todo = [b for b in mine if not block_path(args.out_dir, b).exists()]
    print(f"[task {args.task_id}/{args.n_tasks}] {len(mine)} blocks assigned, {len(todo)} to do, "
          f"{args.workers} workers, rdkit {rdkit.__version__}, host {socket.gethostname()}", flush=True)
    # one flat stream over all pairs of all my unfinished blocks
    flat_block, flat_local, pairs = [], [], []
    for bi, (s, qi, start, end) in enumerate(todo):
        queries, molecules = data[s]
        pairs += [(queries[qi], m) for m in molecules[start:end]]
        flat_block += [bi] * (end - start)
        flat_local += range(end - start)
    arrays, remaining = {}, [b[3] - b[2] for b in todo]
    meta = {"rdkit": rdkit.__version__, "host": socket.gethostname(), "workers": args.workers}
    t_task, n_saved = time.time(), 0

    def on_result(k, res):
        nonlocal n_saved
        bi, j = flat_block[k], flat_local[k]
        a = arrays.setdefault(bi, new_block_arrays(remaining[bi]))
        a["sim"][j], a["tier1"][j], a["tier2"][j], a["status"][j], a["seconds"][j] = res
        if j == len(a["sim"]) - 1:  # results arrive in order: last pair of the block
            save_block(block_path(args.out_dir, todo[bi]), arrays.pop(bi), meta)
            n_saved += 1
            if n_saved % 20 == 0 or n_saved == len(todo):
                print(f"  {n_saved}/{len(todo)} blocks saved | {(time.time() - t_task) / 3600:.2f}h", flush=True)

    holder = PoolHolder(args.workers, args.mem_per_worker_gb)
    try:
        stream_pairs(holder, pairs, on_result)
    finally:
        holder.reset()
    print(f"[task {args.task_id}] finished {len(todo)} blocks in {(time.time() - t_task) / 3600:.2f}h", flush=True)


def cmd_rerun(args):
    """Recompute only the pairs with the given statuses inside finished blocks
    (e.g. hard_fail = memory blow-ups, with a larger --mem_per_worker_gb)."""
    codes = [STATUS_NAMES.index(name) for name in args.status]
    data = load_inputs(args.inputs_dir, args.searches)
    blocks = enumerate_blocks(data)
    mine = [b for idx, b in enumerate(blocks) if idx % args.n_tasks == args.task_id]
    holder = PoolHolder(args.workers, args.mem_per_worker_gb)
    n_pairs = n_fixed = 0
    try:
        for block in mine:
            path = block_path(args.out_dir, block)
            if not path.exists():
                continue
            z = dict(np.load(path))
            idx = np.flatnonzero(np.isin(z["status"], codes))
            if not len(idx):
                continue
            s, qi, start, _ = block
            queries, molecules = data[s]

            def on_result(k, res, z=z, idx=idx):
                i = idx[k]
                z["sim"][i], z["tier1"][i], z["tier2"][i], z["status"][i], z["seconds"][i] = res

            stream_pairs(holder, [(queries[qi], molecules[start + i]) for i in idx], on_result)
            meta = json.loads(str(z.pop("meta")))
            meta.setdefault("reruns", []).append({"statuses": args.status, "n": int(len(idx)), "mem_gb": args.mem_per_worker_gb})
            save_block(path, z, meta)
            fixed = int(np.isin(z["status"][idx], (OK, OK_RETRY)).sum())
            n_pairs += len(idx)
            n_fixed += fixed
            print(f"  search {s} q{qi} mols {start}: reran {len(idx)}, now ok {fixed}", flush=True)
    finally:
        holder.reset()
    print(f"[rerun task {args.task_id}] {n_pairs} pairs recomputed, {n_fixed} now ok")


def cmd_status(args):
    data = load_inputs(args.inputs_dir, args.searches)
    blocks = enumerate_blocks(data)
    for s in args.searches:
        mine = [b for b in blocks if b[0] == s]
        done = sum(block_path(args.out_dir, b).exists() for b in mine)
        print(f"search {s}: {done}/{len(mine)} blocks done ({len(data[s][0])} queries x {len(data[s][1])} molecules)")
    missing = [i for i, b in enumerate(blocks) if not block_path(args.out_dir, b).exists()]
    print(f"total missing blocks: {len(missing)} / {len(blocks)}")


def cmd_combine(args):
    data = load_inputs(args.inputs_dir, args.searches)
    blocks = enumerate_blocks(data)
    for s in args.searches:
        queries, molecules = data[s]
        sim = np.full((len(queries), len(molecules)), np.nan, dtype=np.float32)
        tier1, tier2 = sim.copy(), sim.copy()
        status = np.full(sim.shape, 255, dtype=np.uint8)
        versions, hosts, cpu_seconds, missing = set(), set(), 0.0, 0
        for b in (b for b in blocks if b[0] == s):
            path = block_path(args.out_dir, b)
            if not path.exists():
                missing += 1
                continue
            _, qi, start, end = b
            z = np.load(path)
            sim[qi, start:end], tier1[qi, start:end], tier2[qi, start:end] = z["sim"], z["tier1"], z["tier2"]
            status[qi, start:end] = z["status"]
            meta = json.loads(str(z["meta"]))
            versions.add(meta["rdkit"])
            hosts.add(meta["host"])
            cpu_seconds += float(z["seconds"].sum())
        if missing and not args.allow_missing:
            raise SystemExit(f"search {s}: {missing} blocks missing -- run `status`, resubmit, then combine again")
        d = Path(args.out_dir) / f"search_{s}"
        np.save(d / "gt_sim.npy", sim)
        np.save(d / "gt_tier1.npy", tier1)
        np.save(d / "gt_tier2.npy", tier2)
        np.save(d / "gt_status.npy", status)
        counts = {name: int((status == code).sum()) for code, name in enumerate(STATUS_NAMES)}
        counts["not_computed"] = int((status == 255).sum())
        failed = ~np.isin(status, (OK, OK_RETRY))
        summary = {
            "shape": list(sim.shape),
            "status_counts": counts,
            "nan_fraction": float(np.isnan(sim).mean()),
            "failed_pairs_with_tier1_ge_0.7": int((failed & (tier1 >= RETRY_TIER1)).sum()),
            "queries_with_a_failed_pair_above_their_best_value": int(
                (failed & (tier1 > np.nanmax(np.where(failed, np.nan, sim), axis=1, keepdims=True))).any(axis=1).sum()
            ),
            "rdkit_versions": sorted(versions),
            "hosts": sorted(hosts),
            "cpu_hours": cpu_seconds / 3600,
            "settings": {"threshold": 0.0, "timeout_s": TIMEOUT_1, "retry_timeout_s": TIMEOUT_RETRY,
                         "retry_if_tier1_ge": RETRY_TIER1, "ringMatchesRingOnly": False, "block_size": BLOCK_SIZE},
        }
        (d / "gt_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"search {s}: wrote gt_sim.npy {sim.shape} | {json.dumps(summary['status_counts'])}")


SELFTEST = [
    # known RASCAL bailout pair (a close analog, tier1 0.965): must come back as status 5
    ("CC1OC(OC2C(Oc3c(-c4ccc(O)c(O)c4)oc4cc(OC5OC(CO)C(O)C(O)C5O)cc(O)c4c3=O)OC(CO)C(O)C2O)C(O)C(O)C1O",
     "COc1cc(O)c2c(=O)c(O[C@@H]3O[C@H](CO[C@@H]4O[C@@H](C)[C@H](O)[C@@H](O[C@@H]5O[C@@H](C)[C@H](O)"
     "[C@@H](O)[C@H]5O)[C@H]4O)[C@H](O)[C@H](O)[C@H]3O)c(-c3ccc(O)c(O)c3)oc2c1", BAILOUT, None),
    # ordinary pair: similarity 0.1369 on sofia (rdkit 2023.09.4)
    ("CC1(C)CCc2oc(=O)c3cccc(O)c3c(=O)c2O1",
     "CC[C@H](C)[C@H]1C(=O)O[C@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)O[C@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)"
     "C(=O)O[C@H](C(C)C)C(=O)N1C", OK, 0.136905),
]


def cmd_selftest(args):
    import rdkit

    print(f"rdkit {rdkit.__version__} (sofia reference: 2023.09.4), python {sys.version.split()[0]}")
    with mp.Pool(2, initializer=_init_worker, initargs=(args.mem_per_worker_gb,)) as pool:
        results = pool.map(mces_exact, [(a, b) for a, b, _, _ in SELFTEST])
    ok = True
    for (_, _, want_status, want_sim), (sim, t1, _, st) in zip(SELFTEST, results):
        good = st == want_status and (want_sim is None or abs(sim - want_sim) < 1e-4)
        ok &= good
        print(f"  status {STATUS_NAMES[st]} (want {STATUS_NAMES[want_status]}), sim {sim:.6f}, tier1 {t1:.3f} -> {'OK' if good else 'MISMATCH'}")
    if not ok:
        raise SystemExit("selftest FAILED: different RDKit behaviour -- report back before running")
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selftest", "run", "rerun", "status", "combine"):
        p = sub.add_parser(name)
        p.add_argument("--mem_per_worker_gb", type=float, default=4.0)
        if name == "selftest":
            continue
        p.add_argument("--inputs_dir", required=True)
        p.add_argument("--out_dir", required=True)
        p.add_argument("--searches", nargs="+", default=list(SEARCHES), choices=SEARCHES)
        if name == "rerun":
            p.add_argument("--status", nargs="+", default=["hard_fail"], choices=STATUS_NAMES)
        if name in ("run", "rerun"):
            p.add_argument("--task_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
            p.add_argument("--n_tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
            p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
        if name == "combine":
            p.add_argument("--allow_missing", action="store_true")
    args = ap.parse_args()
    {"selftest": cmd_selftest, "run": cmd_run, "rerun": cmd_rerun, "status": cmd_status, "combine": cmd_combine}[args.cmd](args)


if __name__ == "__main__":
    main()
