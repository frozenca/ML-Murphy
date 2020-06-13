#include <cmath>
#include <vector>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

template <typename N>
concept Unsigned = std::is_unsigned_v<N>;

template <Unsigned U>
constexpr U binom(U N, U K) {
    assert(N >= K);
    U result {1};
    for (U i = U {1}; i <= N - K; i++) {
        result *= i + K;
        result /= i;
    }
    return result;
}

template <Unsigned U>
constexpr double binom_pmf(U N, U K, double theta) {
    assert(theta >= 0.0 && theta <= 1.0);
    return binom<U>(N, K) * std::pow(theta, K) * std::pow(1 - theta, N - K);
}

int main() {
    constexpr size_t N = 15;
    std::vector<size_t> x(N + 1);
    std::vector<double> y(N + 1);

    double theta = 0.25;
    for (size_t K = 0; K <= N; K++) {
        x[K] = K;
        y[K] = binom_pmf<>(N, K, theta);
    }
    // pdf of binomial distribution with n = 15, theta = 0.25
    plt::plot(x, y, "bo");

    plt::show();
}