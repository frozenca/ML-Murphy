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
constexpr F gaussian_pmf(F x, F mean, F stdev) {
    assert(stdev > 0.0);
    return inv_two_sqrtpi * std::exp(-0.5 * std::pow(x - mean, 2) * std::pow(stdev, -2)) / stdev;
}

template <Floating F>
constexpr F student_t_pmf(F x, F mean, F scale, F dof) {
    assert(scale > 0.0 && dof > 0.0);
    return std::tgamma(0.5 + dof / 2.0) / (std::sqrt(dof * M_PI) * std::tgamma(dof / 2.0)) *
           std::pow(1 + std::pow((x - mean) / scale, 2) / dof, -0.5 - dof / 2.0);
}

template <Floating F>
constexpr F laplace_pmf(F x, F mean, F scale) {
    assert(scale > 0.0);
    return std::exp(-std::fabs(x - mean) / scale) / (2.0 * scale);
}

int main() {
    constexpr size_t N = 1000;
    std::vector<double> x(N);
    std::vector<double> y(N);

    double x_start = -4.0;
    double x_end = 4.0;
    for (size_t i = 0; i < N; i++) {
        x[i] = std::lerp(x_start, x_end, static_cast<double>(i) / N);
    }

    double mean = 0.0, stdev = 1.0;
    for (size_t i = 0; i < N; i++) {
        y[i] = gaussian_pmf(x[i], mean, stdev);
    }
    // pdf of N(0, 1)
    plt::plot(x, y, "r");

    mean = 0.0;
    double scale = 1.0, dof = 4.0;
    for (size_t i = 0; i < N; i++) {
        y[i] = student_t_pmf(x[i], mean, scale, dof);
    }
    // pdf of T(0, 1, 4)
    plt::plot(x, y, "b");

    scale = std::pow(2, -0.5);
    for (size_t i = 0; i < N; i++) {
        y[i] = laplace_pmf(x[i], mean, scale);
    }
    // pdf of L(0, 1/sqrt(2))
    plt::plot(x, y, "g");

    plt::show();
}