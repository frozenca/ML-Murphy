#include <cmath>
#include <vector>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

template <typename N>
concept Unsigned = std::is_unsigned_v<N>;

template <typename R>
concept Floating = std::is_floating_point_v<R>;

template <Floating F>
constexpr F pareto_pmf(F x, F k, F m) {
    assert(k > 0.0);
    return (x >= m ? 1.0 : 0.0) * k * std::pow(m, k) * std::pow(x, -k - 1.0);
}

int main() {
    constexpr size_t N = 1000;
    std::vector<double> x(N);
    std::vector<double> y(N);

    double x_start = 0.001;
    double x_end = 4.999;
    for (size_t i = 0; i < N; i++) {
        x[i] = std::lerp(x_start, x_end, static_cast<double>(i) / N);
    }

    double m = 0.01, k = 0.10;
    for (size_t i = 0; i < N; i++) {
        y[i] = pareto_pmf(x[i], k, m);
    }
    // pdf of Pareto(0.01, 0.10)
    plt::plot(x, y, "r");

    m = 0.00, k = 0.50;
    for (size_t i = 0; i < N; i++) {
        y[i] = pareto_pmf(x[i], k, m);
    }
    // pdf of Pareto(0.00, 0.50)
    plt::plot(x, y, "b");

    m = 1.00, k = 1.50;
    for (size_t i = 0; i < N; i++) {
        y[i] = pareto_pmf(x[i], k, m);
    }
    // pdf of Pareto(1.00, 1.00)
    plt::plot(x, y, "g");

    plt::show();
}