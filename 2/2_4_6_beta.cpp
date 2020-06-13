#include <cmath>
#include <vector>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

template <typename N>
concept Unsigned = std::is_unsigned_v<N>;

template <typename R>
concept Floating = std::is_floating_point_v<R>;

template <Floating F>
constexpr F beta_pmf(F x, F a, F b) {
    assert(a > 0.0 && b > 0.0 && x >= 0.0 && x <= 1.0);
    return std::tgamma(a + b) * std::pow(x, a - 1.0) * std::pow(1.0 - x, b - 1.0) / (std::tgamma(a) * std::tgamma(b));
}

int main() {
    constexpr size_t N = 1000;
    std::vector<double> x(N);
    std::vector<double> y(N);

    double x_start = 0.001;
    double x_end = 0.999;
    for (size_t i = 0; i < N; i++) {
        x[i] = std::lerp(x_start, x_end, static_cast<double>(i) / N);
    }

    double a = 0.1, b = 0.1;
    for (size_t i = 0; i < N; i++) {
        y[i] = beta_pmf(x[i], a, b);
    }
    // pdf of Beta(0.1, 0.1)
    plt::plot(x, y, "r");

    a = 1.0, b = 1.0;
    for (size_t i = 0; i < N; i++) {
        y[i] = beta_pmf(x[i], a, b);
    }
    // pdf of Beta(1.0, 1.0)
    plt::plot(x, y, "b");

    a = 2.0, b = 3.0;
    for (size_t i = 0; i < N; i++) {
        y[i] = beta_pmf(x[i], a, b);
    }
    // pdf of Beta(2.0, 3.0)
    plt::plot(x, y, "g");

    a = 8.0, b = 4.0;
    for (size_t i = 0; i < N; i++) {
        y[i] = beta_pmf(x[i], a, b);
    }
    // pdf of Beta(8.0, 4.0)
    plt::plot(x, y, "b-");

    plt::ylim(0.0, 5.0);

    plt::show();
}