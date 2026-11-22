// a small c interface over the cpu reference, so python (ctypes) can call it.
// float32 only. shapes as in lstm.h.
#include "lstm.h"

#include <cstring>

extern "C" {

// h_out [T][B][H]; c_out [T][B][H] may be null
void lstm_cpu_forward(int T, int B, int I, int H, const float* x, const float* w_ih, const float* w_hh,
                      const float* b_ih, const float* b_hh, const float* h0, const float* c0,
                      float* h_out, float* c_out) {
    LstmWeights<float> w(I, H);
    std::memcpy(w.w_ih.data(), w_ih, sizeof(float) * w.w_ih.size());
    std::memcpy(w.w_hh.data(), w_hh, sizeof(float) * w.w_hh.size());
    std::memcpy(w.b_ih.data(), b_ih, sizeof(float) * w.b_ih.size());
    std::memcpy(w.b_hh.data(), b_hh, sizeof(float) * w.b_hh.size());
    LstmCache<float> cache;
    lstm_forward_cpu(w, x, h0, c0, T, B, cache);
    std::memcpy(h_out, cache.h.data(), sizeof(float) * cache.h.size());
    if (c_out) std::memcpy(c_out, cache.c.data(), sizeof(float) * cache.c.size());
}

// gradients for loss = sum(dh_out * h)
void lstm_cpu_backward(int T, int B, int I, int H, const float* x, const float* w_ih, const float* w_hh,
                       const float* b_ih, const float* b_hh, const float* h0, const float* c0,
                       const float* dh_out, float* dx, float* dw_ih, float* dw_hh, float* db_ih,
                       float* db_hh, float* dh0, float* dc0) {
    LstmWeights<float> w(I, H);
    std::memcpy(w.w_ih.data(), w_ih, sizeof(float) * w.w_ih.size());
    std::memcpy(w.w_hh.data(), w_hh, sizeof(float) * w.w_hh.size());
    std::memcpy(w.b_ih.data(), b_ih, sizeof(float) * w.b_ih.size());
    std::memcpy(w.b_hh.data(), b_hh, sizeof(float) * w.b_hh.size());
    LstmCache<float> cache;
    lstm_forward_cpu(w, x, h0, c0, T, B, cache);
    LstmGrads<float> g;
    lstm_backward_cpu(w, x, h0, c0, cache, dh_out, g);
    std::memcpy(dx, g.x.data(), sizeof(float) * g.x.size());
    std::memcpy(dw_ih, g.w_ih.data(), sizeof(float) * g.w_ih.size());
    std::memcpy(dw_hh, g.w_hh.data(), sizeof(float) * g.w_hh.size());
    std::memcpy(db_ih, g.b_ih.data(), sizeof(float) * g.b_ih.size());
    std::memcpy(db_hh, g.b_hh.data(), sizeof(float) * g.b_hh.size());
    std::memcpy(dh0, g.h0.data(), sizeof(float) * g.h0.size());
    std::memcpy(dc0, g.c0.data(), sizeof(float) * g.c0.size());
}
}
