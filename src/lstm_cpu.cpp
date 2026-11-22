// plain loops, no blas. slow, but easy to check against the maths in lstm.h.
#include "lstm.h"

#include <cmath>

namespace {
template <typename F> F sigmoid(F v) { return F(1) / (F(1) + std::exp(-v)); }
}

template <typename F>
void lstm_forward_cpu(const LstmWeights<F>& w, const F* x, const F* h0, const F* c0,
                      int steps, int batch, LstmCache<F>& cache) {
    const int I = w.input, H = w.hidden, G = 4 * H;
    cache.steps = steps;
    cache.batch = batch;
    cache.gates.assign((size_t)steps * batch * G, F(0));
    cache.c.assign((size_t)steps * batch * H, F(0));
    cache.h.assign((size_t)steps * batch * H, F(0));
    std::vector<F> z(G);
    for (int t = 0; t < steps; t++) {
        for (int b = 0; b < batch; b++) {
            const F* xt = x + ((size_t)t * batch + b) * I;
            const F* hp = t ? &cache.h[((size_t)(t - 1) * batch + b) * H] : h0 + (size_t)b * H;
            const F* cp = t ? &cache.c[((size_t)(t - 1) * batch + b) * H] : c0 + (size_t)b * H;
            for (int j = 0; j < G; j++) {
                F s = w.b_ih[j] + w.b_hh[j];
                for (int k = 0; k < I; k++) s += w.w_ih[(size_t)j * I + k] * xt[k];
                for (int k = 0; k < H; k++) s += w.w_hh[(size_t)j * H + k] * hp[k];
                z[j] = s;
            }
            F* gt = &cache.gates[((size_t)t * batch + b) * G];
            F* ct = &cache.c[((size_t)t * batch + b) * H];
            F* ht = &cache.h[((size_t)t * batch + b) * H];
            for (int k = 0; k < H; k++) {
                F i = sigmoid(z[k]), f = sigmoid(z[H + k]), g = std::tanh(z[2 * H + k]), o = sigmoid(z[3 * H + k]);
                gt[k] = i; gt[H + k] = f; gt[2 * H + k] = g; gt[3 * H + k] = o;
                ct[k] = f * cp[k] + i * g;
                ht[k] = o * std::tanh(ct[k]);
            }
        }
    }
}

template <typename F>
void lstm_backward_cpu(const LstmWeights<F>& w, const F* x, const F* h0, const F* c0,
                       const LstmCache<F>& cache, const F* dh_out, LstmGrads<F>& g) {
    const int I = w.input, H = w.hidden, G = 4 * H;
    const int T = cache.steps, B = cache.batch;
    g.x.assign((size_t)T * B * I, F(0));
    g.w_ih.assign(w.w_ih.size(), F(0));
    g.w_hh.assign(w.w_hh.size(), F(0));
    g.b_ih.assign(G, F(0));
    g.b_hh.assign(G, F(0));

    std::vector<F> dh_next((size_t)B * H, F(0)), dc_next((size_t)B * H, F(0)), dz(G);
    for (int t = T - 1; t >= 0; t--) {
        for (int b = 0; b < B; b++) {
            const F* gt = &cache.gates[((size_t)t * B + b) * G];
            const F* ct = &cache.c[((size_t)t * B + b) * H];
            const F* cp = t ? &cache.c[((size_t)(t - 1) * B + b) * H] : c0 + (size_t)b * H;
            const F* hp = t ? &cache.h[((size_t)(t - 1) * B + b) * H] : h0 + (size_t)b * H;
            const F* xt = x + ((size_t)t * B + b) * I;
            F* dhn = &dh_next[(size_t)b * H];
            F* dcn = &dc_next[(size_t)b * H];
            for (int k = 0; k < H; k++) {
                F i = gt[k], f = gt[H + k], gg = gt[2 * H + k], o = gt[3 * H + k];
                F dh = dh_out[((size_t)t * B + b) * H + k] + dhn[k];
                F tc = std::tanh(ct[k]);
                F dc = dcn[k] + dh * o * (F(1) - tc * tc);
                dz[k] = dc * gg * i * (F(1) - i);            // input gate
                dz[H + k] = dc * cp[k] * f * (F(1) - f);     // forget gate
                dz[2 * H + k] = dc * i * (F(1) - gg * gg);   // cell candidate
                dz[3 * H + k] = dh * tc * o * (F(1) - o);    // output gate
                dcn[k] = dc * f;                             // goes to c_{t-1}
            }
            for (int k = 0; k < H; k++) dhn[k] = F(0);
            F* dxt = &g.x[((size_t)t * B + b) * I];
            for (int j = 0; j < G; j++) {
                F d = dz[j];
                g.b_ih[j] += d;
                g.b_hh[j] += d;
                for (int k = 0; k < I; k++) {
                    g.w_ih[(size_t)j * I + k] += d * xt[k];
                    dxt[k] += w.w_ih[(size_t)j * I + k] * d;
                }
                for (int k = 0; k < H; k++) {
                    g.w_hh[(size_t)j * H + k] += d * hp[k];
                    dhn[k] += w.w_hh[(size_t)j * H + k] * d;
                }
            }
        }
    }
    g.h0 = dh_next;
    g.c0 = dc_next;
}

template void lstm_forward_cpu<float>(const LstmWeights<float>&, const float*, const float*, const float*, int, int, LstmCache<float>&);
template void lstm_forward_cpu<double>(const LstmWeights<double>&, const double*, const double*, const double*, int, int, LstmCache<double>&);
template void lstm_backward_cpu<float>(const LstmWeights<float>&, const float*, const float*, const float*, const LstmCache<float>&, const float*, LstmGrads<float>&);
template void lstm_backward_cpu<double>(const LstmWeights<double>&, const double*, const double*, const double*, const LstmCache<double>&, const double*, LstmGrads<double>&);
