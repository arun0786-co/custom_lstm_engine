# custom lstm engine

an lstm layer written from scratch, twice: a plain c++ version that is easy to read and
check, and a cuda version that is fast. both do the forward pass and the full backward
pass (backpropagation through time), with the same maths and weight layout as
`torch.nn.LSTM`, so the results can be compared number for number.

## is it right?

- `tests/test_gradients.cpp` checks every gradient of the c++ version (input, both weight
  matrices, both biases, h0, c0) against finite differences, in double. worst absolute
  error 6e-10.
- `python/compare_torch.py` runs torch.nn.LSTM (in float64) and our version (float32) on
  the same weights and inputs, for the outputs and all gradients. the largest error is
  6e-7 of the largest value, for the c++ and the cuda version alike. that is float32
  rounding.

## is it fast?

forward + backward on an rtx 2080 ti, against `torch.nn.LSTM` (cudnn 9.10, torch 2.14,
fp32), same device tensors, 20 timed runs after 3 warm up runs
(`python/bench_cudnn.py`, `results/bench_2080ti.json`):

| steps T | batch B | input I | hidden H | ours ms | torch / cudnn ms | speed vs torch |
|---|---|---|---|---|---|---|
| 50 | 32 | 128 | 128 | 1.71 | 1.86 | 1.09x |
| 100 | 64 | 256 | 256 | 5.28 | 6.19 | 1.17x |
| 100 | 64 | 512 | 512 | 11.06 | 11.85 | 1.07x |
| 200 | 32 | 512 | 1024 | 31.61 | 33.41 | 1.06x |
| 35 | 20 | 650 | 650 | 3.03 | 3.27 | 1.08x |
| 500 | 8 | 64 | 128 | 10.87 | 2.49 | **0.23x** |

- for medium and big layers it is a little faster than what pytorch gives you. be careful
  with this: the torch time includes autograd's own overhead and the dh0 / dc0 gradients
  are not asked from torch, while we compute them. so "as fast as cudnn through pytorch"
  is the fair summary, not "faster than cudnn".
- for long sequences with a small batch and a small layer it is **4.4x slower**. every
  time step is two kernel launches plus a cublas call each way, and with so little work
  per step the launches are most of the time. cudnn keeps the weights on the chip and
  runs many steps in one persistent kernel. that is the next thing to build here.

## how it works

forward:
1. the input projections of all time steps are one big matrix product,
   `X W_ih^T + b_ih + b_hh` (cublas).
2. per step: `h_{t-1} W_hh^T` is added (cublas), then one fused kernel applies sigmoid /
   tanh, updates c and h and keeps the gate values for the backward pass.

backward:
1. per step, from the last to the first: one fused kernel turns dh and dc into the gate
   gradients dz_t and the dc for the step before; then `dz_t W_hh` gives dh for the step
   before (cublas).
2. after the loop, the weight gradients of all steps are single big products:
   `dW_ih = dZ^T X`, `dW_hh = dZ^T H_prev`, `dX = dZ W_ih`, and the bias gradients are
   column sums of dZ.

the matrix products use cublas. the gate maths is mine (`src/lstm_cuda.cu`).

## run it

```
make test                                   # c++ version + gradient check, any machine
python python/compare_torch.py              # against torch, needs numpy + torch

make cuda                                   # needs nvcc and a gpu
python python/compare_torch.py --cuda
python python/bench_cudnn.py
```

`jobs/ada_cuda.sh` is the slurm job used for the numbers above.

## files

- `include/lstm.h`: the maths and the memory layout, in comments at the top
- `src/lstm_cpu.cpp`: c++ reference, plain loops
- `src/lstm_cuda.cu`: cuda version
- `src/c_api.cpp`: c functions over the c++ version, for python
- `python/`: ctypes wrapper, comparison with torch, benchmark
