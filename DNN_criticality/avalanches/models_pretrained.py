#!/usr/bin/env python3
"""
Avalanche tracker — minimal patch: **--untrained** flag loads random weights.
All other behaviour unchanged.
"""
import os, argparse, math
from contextlib import ExitStack
from functools import lru_cache
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
class StopForward(RuntimeError):
    pass

def _is_leaf(m: nn.Module) -> bool:
    return len(list(m.children())) == 0

@torch.no_grad()
def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            if isinstance(v, torch.Tensor):
                return v
    return None

@torch.no_grad()
def _q_sum(t: torch.Tensor) -> float:
    return float(t.pow(2).sum().item())

# -----------------------------------------------------------------------------
# model builder with optional untrained
# -----------------------------------------------------------------------------

def build_model(name: str, *, untrained: bool = False) -> nn.Module:
    n = name.lower()
    def w(enum):
        return None if untrained else enum

    if n == "resnet18":           return tv.resnet18(weights=w(tv.ResNet18_Weights.DEFAULT))
    if n == "resnet50":           return tv.resnet50(weights=w(tv.ResNet50_Weights.DEFAULT))
    if n == "resnet152":          return tv.resnet152(weights=w(tv.ResNet152_Weights.DEFAULT))
    raise ValueError(f"Unknown model '{name}'")

# -----------------------------------------------------------------------------
# input factory (unchanged)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _model_input_size(model_name: str) -> Tuple[int, int]:
    try:
        _, h, w = getattr(tv, f"{model_name.lower()}_Weights").DEFAULT.meta["input_size"]
        return h, w
    except Exception:
        return 224, 224

def make_input_factory(model_name: str, std: float, device: str = "cpu"):
    H, W = _model_input_size(model_name)
    def make_input():
        return torch.randn(1, 3, H, W, device=device) * std
    return make_input

# -----------------------------------------------------------------------------
# PerLayerThresholdRunner (unchanged logic)
# -----------------------------------------------------------------------------
class PerLayerThresholdRunner:
    def __init__(self, model: nn.Module, theta: List[float],
                 device: Union[str, torch.device] = "cpu",
                 exclude: Iterable[type] = (nn.Identity, nn.Dropout),
                 normalize_avalanche: bool = False):
        self.model = model.to(device).eval()
        self.theta = list(theta)
        self.device = device
        self.exclude = tuple(exclude)
        self.normalize_avalanche = normalize_avalanche
        self.S = 0.0
        self.T = 0
        self._count_above = 0

    def run(self, x: torch.Tensor) -> Tuple[float, int]:
        self.S = 0.0
        self.T = 0
        self._count_above = 0
        x = x.to(self.device)
        step = {"i": 0}

        def post_hook(mod, inputs, output):
            y = _first_tensor(output)
            if y is None:
                return
            j = step["i"]; step["i"] += 1
            if j >= len(self.theta):
                return
            q = _q_sum(y)
            if q >= self.theta[j]:
                r, r_th = math.sqrt(q), math.sqrt(self.theta[j])
                if self.normalize_avalanche:
                    N = float(y.numel())
                    contrib = (r / math.sqrt(N)) - (r_th / math.sqrt(N))
                else:
                    contrib = r - r_th
                if contrib > 0:
                    self.S += contrib
                self._count_above += 1
            else:
                self.T = self._count_above
                raise StopForward

        with torch.no_grad(), ExitStack() as stack:
            for m in self.model.modules():
                if not _is_leaf(m):
                    continue
                if isinstance(m, self.exclude):
                    continue
                stack.enter_context(m.register_forward_hook(post_hook))
            try:
                _ = self.model(x)
            except StopForward:
                pass
        if self.T == 0:
            self.T = self._count_above
        return self.S, self.T

# -----------------------------------------------------------------------------
# calibration (unchanged)
# -----------------------------------------------------------------------------
@torch.no_grad()
def calibrate_mean_std_qsum(model, make_input, K, device="cpu", exclude=(nn.Identity, nn.Dropout)):
    q_lists = []
    for _ in range(K):
        q = []
        def hook(_, __, out):
            t = _first_tensor(out)
            if t is not None:
                q.append(_q_sum(t))
        with ExitStack() as stack:
            for m in model.modules():
                if not _is_leaf(m): continue
                if isinstance(m, exclude): continue
                stack.enter_context(m.register_forward_hook(hook))
            _ = model(make_input().to(device))
        q_lists.append(q)
    L = min(len(q) for q in q_lists)
    qs = torch.tensor([q[:L] for q in q_lists], dtype=torch.float64)
    mu = qs.mean(0)
    sd = qs.std(0, unbiased=True)
    return mu, sd

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--std", type=float, required=True)
    ap.add_argument("--calib_runs", type=int, required=True)
    ap.add_argument("--runs", type=int, required=True)
    ap.add_argument("--include_dropout", action="store_true")
    ap.add_argument("--use_batch_stats", action="store_true")
    ap.add_argument("--normalize_avalanche", action="store_true")
    ap.add_argument("--out_dir", default="results_pretrained")
    ap.add_argument("--untrained", action="store_true",
                    help="Load the model with random weights (weights=None)")
    ap.add_argument("--out_file", type=str, default=None,
                    help="Optional exact name of .npy file to save (inside --out_dir)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    model = build_model(args.model, untrained=args.untrained).to(device).eval()
    exclude = (nn.Identity,) if args.include_dropout else (nn.Identity, nn.Dropout)
    make_input = make_input_factory(args.model, args.std, device)

    # optional BN train mode
    saved_bn = []
    if args.use_batch_stats:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                saved_bn.append((m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone(), m.training))
                m.train()

    mu, sd = calibrate_mean_std_qsum(model, make_input, K=args.calib_runs, device=device, exclude=exclude)
    theta = mu + 0*sd # θ = μ

    runner = PerLayerThresholdRunner(model, theta, device=device, exclude=exclude, normalize_avalanche=args.normalize_avalanche)
    s_list, t_list = [], []
    for i in range(args.runs):
        S, T = runner.run(make_input())
        if i % 1000 == 0:
            print(f"step {i:7d} | {args.model} | std{args.std:.5f} | T={T:4d} | S={S:.6f}")
        s_list.append(S); t_list.append(T)

    for m, rm, rv, nbt, was_train in saved_bn:
        m.running_mean.copy_(rm); m.running_var.copy_(rv)
        m.num_batches_tracked.copy_(nbt); m.train(was_train)

    if args.out_file:
        out_path = os.path.join(args.out_dir, args.out_file)
    else:
        tag = f"{args.model}_std{args.std:.5f}_K{args.calib_runs}_N{args.runs}" + (
            "_untrained" if args.untrained else "_pretrained")
        out_path = os.path.join(args.out_dir, f"avalanches_{tag}_thresh2.5std.npy")

    np.save(out_path, (np.array(s_list, dtype=np.float64), np.array(t_list, dtype=np.int64)))
    print("Saved", out_path)

if __name__ == "__main__":
    main()