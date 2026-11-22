// lstm forward and backward on the gpu.
//
// the matrix products go to cublas; the gate maths is two fused kernels of
// my own (one for the forward step, one for the backward step).
//
// forward:  xg = X W_ih^T + b_ih + b_hh for all steps at once (one big gemm),
//           then per step: z = xg_t + h_{t-1} W_hh^T (gemm), then the gate kernel.
// backward: per step: gate backward kernel -> dz_t, then dh_{t-1} = dz_t W_hh (gemm).
//           after the loop the weight grads are single big gemms over all steps.
//
// cublas is column major. a row major [m][n] matrix is the same memory as a
// column major n x m one, so row major C = A B is computed as C^T = B^T A^T.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_OK(e) do { cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::fprintf(stderr, "cuda error %s at %s:%d\n", cudaGetErrorString(r_), __FILE__, __LINE__); std::abort(); } } while (0)
#define BLAS_OK(e) do { cublasStatus_t r_ = (e); if (r_ != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cublas error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); std::abort(); } } while (0)

namespace {

cublasHandle_t handle() {
    static cublasHandle_t h = nullptr;
    if (!h) BLAS_OK(cublasCreate(&h));
    return h;
}

// row major: C[m][n] = alpha * op(A) op(B) + beta * C, with op(A) [m][k], op(B) [k][n].
// ta / tb say whether A / B are stored transposed (A as [k][m], B as [n][k]).
void gemm(bool ta, bool tb, int m, int n, int k, float alpha, const float* A, const float* B,
          float beta, float* C) {
    int lda = ta ? m : k, ldb = tb ? k : n;
    BLAS_OK(cublasSgemm(handle(), tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
                        n, m, k, &alpha, B, ldb, A, lda, &beta, C, n));
}

__device__ float sigmoidf(float v) { return 1.f / (1.f + __expf(-v)); }

// z [B][4H] pre-activations for one step (overwritten by the activated gates),
// c_prev, c, h [B][H]
__global__ void gates_forward(int B, int H, float* z, const float* c_prev, float* c, float* h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    int b = idx / H, k = idx % H;
    float* zb = z + (size_t)b * 4 * H;
    float i = sigmoidf(zb[k]), f = sigmoidf(zb[H + k]), g = tanhf(zb[2 * H + k]), o = sigmoidf(zb[3 * H + k]);
    zb[k] = i; zb[H + k] = f; zb[2 * H + k] = g; zb[3 * H + k] = o;
    float cv = f * c_prev[idx] + i * g;
    c[idx] = cv;
    h[idx] = o * tanhf(cv);
}

// one backward step. dh = dh_out_t + dh_next (dh_next already holds dz_{t+1} W_hh).
// writes dz [B][4H] and updates dc in place (dc_t -> dc_{t-1}).
__global__ void gates_backward(int B, int H, const float* gates, const float* c, const float* c_prev,
                               const float* dh_out, const float* dh_next, float* dc, float* dz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    int b = idx / H, k = idx % H;
    const float* gb = gates + (size_t)b * 4 * H;
    float i = gb[k], f = gb[H + k], g = gb[2 * H + k], o = gb[3 * H + k];
    float dh = dh_out[idx] + dh_next[idx];
    float tc = tanhf(c[idx]);
    float dcv = dc[idx] + dh * o * (1.f - tc * tc);
    float* zb = dz + (size_t)b * 4 * H;
    zb[k] = dcv * g * i * (1.f - i);
    zb[H + k] = dcv * c_prev[idx] * f * (1.f - f);
    zb[2 * H + k] = dcv * i * (1.f - g * g);
    zb[3 * H + k] = dh * tc * o * (1.f - o);
    dc[idx] = dcv * f;
}

// out[j] = sum over rows of m [rows][cols] (bias gradients)
__global__ void column_sum(int rows, int cols, const float* m, float* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.f;
    for (int r = 0; r < rows; r++) s += m[(size_t)r * cols + j];
    out[j] = s;
}

__global__ void add_bias(int rows, int cols, float* m, const float* b1, const float* b2) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)rows * cols) return;
    int j = idx % cols;
    m[idx] += b1[j] + b2[j];
}

int blocks(size_t n, int t = 256) { return (int)((n + t - 1) / t); }

}  // namespace

extern "C" {

// device pointers. gates [T][B][4H], c and h [T][B][H] are outputs and must be
// kept for the backward pass.
void lstm_cuda_forward_dev(int T, int B, int I, int H, const float* x, const float* w_ih, const float* w_hh,
                           const float* b_ih, const float* b_hh, const float* h0, const float* c0,
                           float* gates, float* c, float* h) {
    const int G = 4 * H;
    // all input projections at once, straight into the gates buffer
    gemm(false, true, T * B, G, I, 1.f, x, w_ih, 0.f, gates);
    add_bias<<<blocks((size_t)T * B * G), 256>>>(T * B, G, gates, b_ih, b_hh);
    for (int t = 0; t < T; t++) {
        float* zt = gates + (size_t)t * B * G;
        const float* hp = t ? h + (size_t)(t - 1) * B * H : h0;
        const float* cp = t ? c + (size_t)(t - 1) * B * H : c0;
        gemm(false, true, B, G, H, 1.f, hp, w_hh, 1.f, zt);
        gates_forward<<<blocks((size_t)B * H), 256>>>(B, H, zt, cp, c + (size_t)t * B * H, h + (size_t)t * B * H);
    }
    CUDA_OK(cudaGetLastError());
}

// device pointers. dz is a [T][B][4H] work buffer, hprev a [T][B][H] work buffer.
// dh0 and dc0 [B][H] are outputs (dc0 is also used as the running dc).
void lstm_cuda_backward_dev(int T, int B, int I, int H, const float* x, const float* w_ih, const float* w_hh,
                            const float* h0, const float* c0, const float* gates, const float* c,
                            const float* h, const float* dh_out, float* dz, float* hprev, float* dx,
                            float* dw_ih, float* dw_hh, float* db_ih, float* db_hh, float* dh0, float* dc0) {
    const int G = 4 * H;
    const size_t BH = (size_t)B * H;
    float* dh_next = dh0;  // reused: holds dz_{t+1} W_hh, ends as dh0
    float* dc = dc0;       // running dc, ends as dc0
    CUDA_OK(cudaMemsetAsync(dh_next, 0, BH * sizeof(float)));
    CUDA_OK(cudaMemsetAsync(dc, 0, BH * sizeof(float)));
    for (int t = T - 1; t >= 0; t--) {
        const float* cp = t ? c + (size_t)(t - 1) * BH : c0;
        float* dzt = dz + (size_t)t * B * G;
        gates_backward<<<blocks(BH), 256>>>(B, H, gates + (size_t)t * B * G, c + (size_t)t * BH, cp,
                                            dh_out + (size_t)t * BH, dh_next, dc, dzt);
        gemm(false, false, B, H, G, 1.f, dzt, w_hh, 0.f, dh_next);
    }
    // h_{t-1} for every step: h0 then h[0..T-2]
    CUDA_OK(cudaMemcpyAsync(hprev, h0, BH * sizeof(float), cudaMemcpyDeviceToDevice));
    if (T > 1) CUDA_OK(cudaMemcpyAsync(hprev + BH, h, (T - 1) * BH * sizeof(float), cudaMemcpyDeviceToDevice));
    gemm(true, false, G, I, T * B, 1.f, dz, x, 0.f, dw_ih);      // dW_ih = dZ^T X
    gemm(true, false, G, H, T * B, 1.f, dz, hprev, 0.f, dw_hh);  // dW_hh = dZ^T H_prev
    gemm(false, false, T * B, I, G, 1.f, dz, w_ih, 0.f, dx);     // dX = dZ W_ih
    column_sum<<<blocks(G), 256>>>(T * B, G, dz, db_ih);
    CUDA_OK(cudaMemcpyAsync(db_hh, db_ih, G * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_OK(cudaGetLastError());
}

// host pointer versions with the same arguments as the cpu c api (for tests)
static float* up(const float* src, size_t n) {
    float* d;
    CUDA_OK(cudaMalloc(&d, n * sizeof(float)));
    if (src) CUDA_OK(cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}
static void down(float* dst, const float* d, size_t n) {
    if (dst) CUDA_OK(cudaMemcpy(dst, d, n * sizeof(float), cudaMemcpyDeviceToHost));
}

void lstm_cuda_forward(int T, int B, int I, int H, const float* x, const float* w_ih, const float* w_hh,
                       const float* b_ih, const float* b_hh, const float* h0, const float* c0,
                       float* h_out, float* c_out) {
    size_t G = 4 * (size_t)H, TB = (size_t)T * B;
    float *dx = up(x, TB * I), *wi = up(w_ih, G * I), *wh = up(w_hh, G * H), *bi = up(b_ih, G),
          *bh = up(b_hh, G), *dh0 = up(h0, (size_t)B * H), *dc0 = up(c0, (size_t)B * H),
          *gates = up(nullptr, TB * G), *c = up(nullptr, TB * H), *h = up(nullptr, TB * H);
    lstm_cuda_forward_dev(T, B, I, H, dx, wi, wh, bi, bh, dh0, dc0, gates, c, h);
    down(h_out, h, TB * H);
    down(c_out, c, TB * H);
    for (float* p : {dx, wi, wh, bi, bh, dh0, dc0, gates, c, h}) cudaFree(p);
}

void lstm_cuda_backward(int T, int B, int I, int H, const float* x, const float* w_ih, const float* w_hh,
                        const float* b_ih, const float* b_hh, const float* h0, const float* c0,
                        const float* dh_out, float* gx, float* gw_ih, float* gw_hh, float* gb_ih,
                        float* gb_hh, float* gh0, float* gc0) {
    size_t G = 4 * (size_t)H, TB = (size_t)T * B, BH = (size_t)B * H;
    float *dx = up(x, TB * I), *wi = up(w_ih, G * I), *wh = up(w_hh, G * H), *bi = up(b_ih, G),
          *bh = up(b_hh, G), *dh0 = up(h0, BH), *dc0 = up(c0, BH), *gates = up(nullptr, TB * G),
          *c = up(nullptr, TB * H), *h = up(nullptr, TB * H), *ddh = up(dh_out, TB * H),
          *dz = up(nullptr, TB * G), *hprev = up(nullptr, TB * H), *ox = up(nullptr, TB * I),
          *owi = up(nullptr, G * I), *owh = up(nullptr, G * H), *obi = up(nullptr, G), *obh = up(nullptr, G),
          *oh0 = up(nullptr, BH), *oc0 = up(nullptr, BH);
    lstm_cuda_forward_dev(T, B, I, H, dx, wi, wh, bi, bh, dh0, dc0, gates, c, h);
    lstm_cuda_backward_dev(T, B, I, H, dx, wi, wh, dh0, dc0, gates, c, h, ddh, dz, hprev, ox, owi, owh, obi,
                           obh, oh0, oc0);
    down(gx, ox, TB * I); down(gw_ih, owi, G * I); down(gw_hh, owh, G * H); down(gb_ih, obi, G);
    down(gb_hh, obh, G); down(gh0, oh0, BH); down(gc0, oc0, BH);
    for (float* p : {dx, wi, wh, bi, bh, dh0, dc0, gates, c, h, ddh, dz, hprev, ox, owi, owh, obi, obh, oh0, oc0})
        cudaFree(p);
}
}
