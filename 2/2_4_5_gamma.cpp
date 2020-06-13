#include <cmath>
#include <vector>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

// FIXME: use C++20 std::inv_sqrtpi_v, std::sqrt2_v when implemented
static const double inv_two_sqrtpi = std::pow(2.0 * M_PI, -0.5);

template <typename N>
concept Unsigned = std::is_unsigned_v<N>;

template <typename R>
concept Floating = std::is_floating_point_v<R>;

template <Floating F>
constexpr F gamma_pmf(F x, F shape, F rate) {
    assert(shape > 0.0 && rate > 0.0 && x > 0.0);
    return std::pow(rate, shape) * std::pow(x, shape - 1) * std::exp(-x * rate) / std::tgamma(shape);
}

int main() {
    constexpr size_t N = 1000;
    std::vector<double> x(N);
    std::vector<double> y(N);

    double x_start = 0.001;
    double x_end = 4.0;
    for (size_t i = 0; i < N; i++) {
        x[i] = std::lerp(x_start, x_end, static_cast<double>(i) / N);
    }

    double a = 1.0, b = 1.0;
    for (size_t i = 0; i < N; i++) {
        y[i] = gamma_pmf(x[i], a, b);
    }
    // pdf of Gamma(1.0, 1.0)
    plt::plot(x, y, "r");

    a = 1.5, b = 1.0;
    for (size_t i = 0; i < N; i++) {
        y[i] = gamma_pmf(x[i], a, b);
    }
    // pdf of Gamma(1.5, 1.0)
    plt::plot(x, y, "b");

    a = 2.0, b = 1.0;
    for (size_t i = 0; i < N; i++) {
        y[i] = gamma_pmf(x[i], a, b);
    }
    // pdf of Gamma(1.5, 1.0)
    plt::plot(x, y, "g");

    plt::show();
}