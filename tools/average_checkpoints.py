"""Average the model weights of several Lightning checkpoints of one run
(e.g. the diag_step_*.ckpt files saved by the training diagnostics).

Floating-point tensors of state_dict are averaged; all other entries (integer
buffers, hyperparameters, optimizer state, ...) are taken from the last
checkpoint given. The result loads like any SIMBA checkpoint.

Usage:
  .venv/bin/python tools/average_checkpoints.py --out avg_last5.ckpt ckpt1 ckpt2 ...
  .venv/bin/python tools/average_checkpoints.py --out avg_last5.ckpt --last 5 <run>/diag_checkpoints
"""

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="checkpoint files, or one directory of diag_step_*.ckpt")
    ap.add_argument("--last", type=int, default=None, help="with a directory: average the last N checkpoints by step")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    if len(paths) == 1 and paths[0].is_dir():
        paths = sorted(paths[0].glob("diag_step_*.ckpt"))
        if args.last:
            paths = paths[-args.last:]
    if not paths:
        raise SystemExit("no checkpoints found")

    total, last = None, None
    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["state_dict"]
        if total is None:
            total = {k: v.double().clone() if v.is_floating_point() else None for k, v in state.items()}
        else:
            for k, v in state.items():
                if total[k] is not None:
                    total[k] += v.double()
        last = ckpt
    averaged = {k: (total[k] / len(paths)).to(v.dtype) if total[k] is not None else v
                for k, v in last["state_dict"].items()}
    last["state_dict"] = averaged
    last["averaged_from"] = [str(p) for p in paths]
    torch.save(last, args.out)
    print(f"averaged {len(paths)} checkpoints ({paths[0].name} .. {paths[-1].name}) -> {args.out}")


if __name__ == "__main__":
    main()
