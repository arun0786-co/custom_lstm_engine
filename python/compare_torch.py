"""compare our lstm with torch.nn.LSTM on the same weights and inputs.

  python python/compare_torch.py            # cpu reference
  python python/compare_torch.py --cuda     # cuda version (on a gpu machine)
"""
import argparse

import numpy as np
import torch

from lstm_lib import backward, forward, load


def run(lib, prefix, T, B, I, H, seed=0):
    torch.manual_seed(seed)
    ref = torch.nn.LSTM(I, H).double()
    x = torch.randn(T, B, I, dtype=torch.float64, requires_grad=True)
    h0 = torch.randn(1, B, H, dtype=torch.float64, requires_grad=True)
    c0 = torch.randn(1, B, H, dtype=torch.float64, requires_grad=True)
    out, _ = ref(x, (h0, c0))
    dh = torch.randn_like(out)
    (out * dh).sum().backward()

    f = lambda t: np.ascontiguousarray(t.detach().numpy().astype(np.float32))
    args = [f(x), f(ref.weight_ih_l0), f(ref.weight_hh_l0), f(ref.bias_ih_l0), f(ref.bias_hh_l0), f(h0[0]), f(c0[0])]
    h, _ = forward(lib, *args, prefix=prefix)
    g = backward(lib, *args, f(dh), prefix=prefix)
    want = {"h": out, "x": x.grad, "w_ih": ref.weight_ih_l0.grad, "w_hh": ref.weight_hh_l0.grad,
            "b_ih": ref.bias_ih_l0.grad, "b_hh": ref.bias_hh_l0.grad, "h0": h0.grad[0], "c0": c0.grad[0]}
    got = {"h": h, **g}
    worst = 0.0
    for k, v in want.items():
        v = v.detach().numpy()
        err = np.abs(got[k] - v).max() / (np.abs(v).max() + 1e-12)
        worst = max(worst, err)
        print(f"  {k:5s} max error / max value {err:.1e}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    lib = load("liblstm_cuda" if a.cuda else "liblstm_cpu")
    prefix = "lstm_cuda" if a.cuda else "lstm_cpu"
    bad = 0
    for T, B, I, H in [(1, 1, 3, 4), (7, 3, 5, 8), (20, 16, 32, 64)]:
        print(f"T={T} B={B} I={I} H={H}")
        w = run(lib, prefix, T, B, I, H)
        ok = w < 1e-4  # float32 against a float64 reference
        bad += not ok
        print("  ok" if ok else "  FAIL")
    raise SystemExit(1 if bad else 0)
