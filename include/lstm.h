// one lstm layer, same maths and layout as torch.nn.LSTM (gate order i, f, g, o).
//
//   z   = W_ih x_t + b_ih + W_hh h_{t-1} + b_hh        (4H values)
//   i   = sigmoid(z[0:H])      f = sigmoid(z[H:2H])
//   g   = tanh(z[2H:3H])       o = sigmoid(z[3H:4H])
//   c_t = f * c_{t-1} + i * g
//   h_t = o * tanh(c_t)
//
// all arrays are row major and time major:
//   x  [T][B][I]    h, c  [T][B][H]    W_ih [4H][I]    W_hh [4H][H]
#pragma once
#include <cstddef>
#include <vector>

template <typename F>
struct LstmWeights {
    int input = 0, hidden = 0;
    std::vector<F> w_ih, w_hh, b_ih, b_hh;  // [4H][I], [4H][H], [4H], [4H]
    LstmWeights(int i, int h)
        : input(i), hidden(h), w_ih(4 * h * i), w_hh(4 * h * h), b_ih(4 * h), b_hh(4 * h) {}
};

// what the forward pass keeps for the backward pass
template <typename F>
struct LstmCache {
    int steps = 0, batch = 0;
    std::vector<F> gates;  // [T][B][4H] after the activations: i, f, g, o
    std::vector<F> c;      // [T][B][H]
    std::vector<F> h;      // [T][B][H]  (this is also the output)
};

template <typename F>
struct LstmGrads {
    std::vector<F> x, w_ih, w_hh, b_ih, b_hh, h0, c0;
};

// cpu reference, in src/lstm_cpu.cpp. h0 and c0 are [B][H].
template <typename F>
void lstm_forward_cpu(const LstmWeights<F>& w, const F* x, const F* h0, const F* c0,
                      int steps, int batch, LstmCache<F>& cache);

// dh_out [T][B][H] is the gradient of the loss with respect to every h_t.
template <typename F>
void lstm_backward_cpu(const LstmWeights<F>& w, const F* x, const F* h0, const F* c0,
                       const LstmCache<F>& cache, const F* dh_out, LstmGrads<F>& g);
