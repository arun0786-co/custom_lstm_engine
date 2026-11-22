"""time forward + backward of our cuda lstm against torch.nn.LSTM (cudnn).

both get the same device tensors; our library is called through its device
pointer api, so no copies are timed.

  python python/bench_cudnn.py --out results/bench_2080ti.json
"""
import argparse
import ctypes
import json
import time

import torch

from lstm_lib import load

P = ctypes.c_void_p


def p(t):
    return P(t.data_ptr())


def ours(lib, x, w_ih, w_hh, b_ih, b_hh, h0, c0, dh, buf):
    T, B, I = x.shape
    H = h0.shape[-1]
    lib.lstm_cuda_forward_dev(T, B, I, H, *map(p, (x, w_ih, w_hh, b_ih, b_hh, h0, c0, buf["gates"], buf["c"], buf["h"])))
    lib.lstm_cuda_backward_dev(T, B, I, H, *map(p, (x, w_ih, w_hh, h0, c0, buf["gates"], buf["c"], buf["h"], dh,
                                                   buf["dz"], buf["hprev"], buf["dx"], buf["dw_ih"], buf["dw_hh"],
                                                   buf["db_ih"], buf["db_hh"], buf["dh0"], buf["dc0"])))


def timeit(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lib = load("liblstm_cuda")
    dev = "cuda"
    rows = []
    for T, B, I, H in [(50, 32, 128, 128), (100, 64, 256, 256), (100, 64, 512, 512), (200, 32, 512, 1024),
                       (35, 20, 650, 650), (500, 8, 64, 128)]:
        torch.manual_seed(0)
        ref = torch.nn.LSTM(I, H).to(dev)
        x = torch.randn(T, B, I, device=dev)
        h0 = torch.randn(1, B, H, device=dev)
        c0 = torch.randn(1, B, H, device=dev)
        dh = torch.randn(T, B, H, device=dev)
        xr = x.clone().requires_grad_(True)

        def cudnn():
            ref.zero_grad(set_to_none=True)
            out, _ = ref(xr, (h0, c0))
            out.backward(dh)

        G = 4 * H
        e = lambda *s: torch.empty(*s, device=dev)
        buf = {"gates": e(T, B, G), "c": e(T, B, H), "h": e(T, B, H), "dz": e(T, B, G), "hprev": e(T, B, H),
               "dx": e(T, B, I), "dw_ih": e(G, I), "dw_hh": e(G, H), "db_ih": e(G), "db_hh": e(G),
               "dh0": e(B, H), "dc0": e(B, H)}
        args = (x, ref.weight_ih_l0.detach(), ref.weight_hh_l0.detach(), ref.bias_ih_l0.detach(),
                ref.bias_hh_l0.detach(), h0[0].contiguous(), c0[0].contiguous(), dh, buf)

        # same answer first
        ours(lib, *args)
        cudnn()
        torch.cuda.synchronize()
        err_h = ((buf["h"] - ref(x, (h0, c0))[0]).abs().max() / buf["h"].abs().max()).item()
        err_w = ((buf["dw_hh"] - ref.weight_hh_l0.grad).abs().max() / ref.weight_hh_l0.grad.abs().max()).item()

        reps = 20
        t_ours = timeit(lambda: ours(lib, *args), reps)
        t_cudnn = timeit(cudnn, reps)
        row = {"T": T, "B": B, "I": I, "H": H, "ours_ms": round(t_ours, 3), "cudnn_ms": round(t_cudnn, 3),
               "ours_vs_cudnn": round(t_cudnn / t_ours, 3), "max_rel_err_h": err_h, "max_rel_err_dw_hh": err_w}
        rows.append(row)
        print(row, flush=True)
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cudnn": torch.backends.cudnn.version(), "rows": rows}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
