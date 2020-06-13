#include <cmath>
#include <vector>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

// FIXME: use C++20 std::inv_sqrtpi_v, std::sqrt2_v when implemented
static const double inv_two_sqrtpi = std::pow(2.0 * M_PI, -0.5);

int main() {
    size_t N = 10, N2 = 1'000;
    std::vector<double> x(N + 1), y(N + 1), x2(N2);

    double x_start = -5;
    double x_end = 5;

    // Uniform distribution
    for (size_t i = 0; i <= N; i++) {
        x[i] = std::lerp(x_start, x_end, static_cast<double>(i) / N);
        y[i] = 1.0 / (x_end - x_start);
    }
    plt::plot(x, y, "bo");

    // Standard normal distribution
    for (size_t i = 0; i < N2; i++) {
        x2[i] = std::lerp(x_start, x_end, static_cast<double>(i) / N2);
    }

    plt::plot(x2, [](double d) {return inv_two_sqrtpi * std::exp(-0.5 * std::pow(d, 2));}, "r");
    plt::show();
}