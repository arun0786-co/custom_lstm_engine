// checks the cpu backward pass against finite differences, in double.
// loss = sum over all h_t of r * h_t, with fixed random r, so dh_out = r.
#include "lstm.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>

static int failures = 0;

static double loss_of(const LstmWeights<double>& w, const std::vector<double>& x, const std::vector<double>& h0,
                      const std::vector<double>& c0, const std::vector<double>& r, int T, int B) {
    LstmCache<double> cache;
    lstm_forward_cpu(w, x.data(), h0.data(), c0.data(), T, B, cache);
    double s = 0;
    for (size_t i = 0; i < r.size(); i++) s += r[i] * cache.h[i];
    return s;
}

static void check(const char* name, std::vector<double>& param, const std::vector<double>& grad,
                  const std::function<double()>& loss) {
    const double eps = 1e-6;
    // pass if every |numeric - analytic| <= 1e-8 + 1e-5 * |numeric|. a pure relative
    // test fails on tiny gradients, where finite differences only get ~1e-10 right.
    double worst = 0, worst_abs = 0, at_grad = 0;
    bool ok = true;
    for (size_t i = 0; i < param.size(); i++) {
        double keep = param[i];
        param[i] = keep + eps; double up = loss();
        param[i] = keep - eps; double down = loss();
        param[i] = keep;
        double num = (up - down) / (2 * eps);
        double rel = std::fabs(num - grad[i]) / std::fmax(1e-8, std::fabs(num) + std::fabs(grad[i]));
        if (rel > worst) { worst = rel; at_grad = grad[i]; }
        worst_abs = std::fmax(worst_abs, std::fabs(num - grad[i]));
        if (std::fabs(num - grad[i]) > 1e-8 + 1e-5 * std::fabs(num)) ok = false;
    }
    if (!ok) failures++;
    std::printf("%-5s %-6s worst relative error %.2e (at a gradient of %.1e), worst abs error %.1e, %zu values\n",
                ok ? "ok" : "FAIL", name, worst, at_grad, worst_abs, param.size());
}

int main() {
    const int T = 5, B = 3, I = 4, H = 6;
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> u(-0.8, 0.8);
    LstmWeights<double> w(I, H);
    for (auto* v : {&w.w_ih, &w.w_hh, &w.b_ih, &w.b_hh}) for (auto& e : *v) e = u(rng);
    std::vector<double> x(T * B * I), h0(B * H), c0(B * H), r(T * B * H);
    for (auto* v : {&x, &h0, &c0, &r}) for (auto& e : *v) e = u(rng);

    LstmCache<double> cache;
    lstm_forward_cpu(w, x.data(), h0.data(), c0.data(), T, B, cache);
    LstmGrads<double> g;
    lstm_backward_cpu(w, x.data(), h0.data(), c0.data(), cache, r.data(), g);

    auto loss = [&] { return loss_of(w, x, h0, c0, r, T, B); };
    check("x", x, g.x, loss);
    check("w_ih", w.w_ih, g.w_ih, loss);
    check("w_hh", w.w_hh, g.w_hh, loss);
    check("b_ih", w.b_ih, g.b_ih, loss);
    check("b_hh", w.b_hh, g.b_hh, loss);
    check("h0", h0, g.h0, loss);
    check("c0", c0, g.c0, loss);
    return failures ? 1 : 0;
}
