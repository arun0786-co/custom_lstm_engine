"""ctypes wrapper around build/liblstm_cpu.so (and later the cuda library)."""
import ctypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P = ctypes.POINTER(ctypes.c_float)


def load(name="liblstm_cpu"):
    for ext in (".so", ".dylib"):
        f = ROOT / "build" / (name + ext)
        if f.exists():
            return ctypes.CDLL(str(f))
    raise FileNotFoundError(f"build/{name}.so not found, run make first")


def ptr(a):
    assert a.dtype == np.float32 and a.flags["C_CONTIGUOUS"]
    return a.ctypes.data_as(P)


def forward(lib, x, w_ih, w_hh, b_ih, b_hh, h0, c0, prefix="lstm_cpu"):
    T, B, I = x.shape
    H = h0.shape[-1]
    h = np.zeros((T, B, H), np.float32)
    c = np.zeros((T, B, H), np.float32)
    getattr(lib, prefix + "_forward")(T, B, I, H, *map(ptr, (x, w_ih, w_hh, b_ih, b_hh, h0, c0, h, c)))
    return h, c


def backward(lib, x, w_ih, w_hh, b_ih, b_hh, h0, c0, dh, prefix="lstm_cpu"):
    T, B, I = x.shape
    H = h0.shape[-1]
    out = [np.zeros_like(a) for a in (x, w_ih, w_hh, b_ih, b_hh, h0, c0)]
    getattr(lib, prefix + "_backward")(T, B, I, H, *map(ptr, (x, w_ih, w_hh, b_ih, b_hh, h0, c0, dh)),
                                       *map(ptr, out))
    return dict(zip(["x", "w_ih", "w_hh", "b_ih", "b_hh", "h0", "c0"], out))
