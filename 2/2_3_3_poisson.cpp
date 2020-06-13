#include <cmath>
#include <vector>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

template <typename N>
concept Unsigned = std::is_unsigned_v<N>;

template <typename R>
concept Floating = std::is_floating_point_v<R>;


template <Unsigned U, Floating F>
constexpr F poisson_pmf(U N, F lambda) {
    assert(lambda > 0.0);
    return std::exp(-lambda) * std::pow(lambda, N) / std::tgamma(N + 1);
}

int main() {
    constexpr size_t N = 15;
    std::vector<size_t> x(N + 1);
    std::vector<double> y(N + 1);

    double theta = 0.25;
    for (size_t K = 0; K <= N; K++) {
        x[K] = K;
        y[K] = poisson_pmf<>(K, theta);
    }
    // pdf of binomial distribution with n = 15, theta = 0.25
    plt::plot(x, y, "bo");

    plt::show();
}