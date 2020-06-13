#include <algorithm>
#include <cmath>
#include <cassert>
#include <utility>
#include <initializer_list>
#include <iostream>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

template <typename N>
concept Unsigned = std::is_unsigned_v<N>;

template <typename N>
concept Integral = std::is_integral_v<N>;

template <typename R>
concept Floating = std::is_floating_point_v<R>;

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

template <Unsigned U, Integral I>
constexpr U multinom (U N, I K) {
    assert(N == K);
    return U {1};
}

template <Unsigned U, Integral I, Integral... Is>
constexpr U multinom (U N, I K1, Is... Ks) {
    assert(N >= static_cast<U>(K1));
    return binom(N, static_cast<U>(K1)) * multinom(N - static_cast<U>(K1), Ks...);
}

template <Floating F1, Floating F2>
constexpr bool almost_equal(F1 f1, F2 f2) {
    return std::fabs(f1 - f2) < std::min({1.0e-4, f1 * 1.0e-3, f2 * 1.0e-3});
}

template <Unsigned U, Integral I, Floating F, typename... Ts>
constexpr F multinom_pmf (U N, I K1, F theta1, Ts... args) {
    assert(theta1 >= 0.0);
    F prob_value = std::pow(theta1, K1) * binom(N, static_cast<U>(K1));
    if constexpr(sizeof...(args) > 0)
        return prob_value * multinom_pmf(N - static_cast<U>(K1), args...);
    else
        return prob_value;
}

int main() {
    constexpr size_t N = 15;
    std::vector<double> x(N + 1), y(N + 1);
    std::vector<double> z(N + 1);

    double theta = 0.3;
    for (size_t i = 0; i <= N; i++) {
        x[i] = i;
        y[i] = N - i;
        z[i] = multinom_pmf(N, i, theta, N - i, 1.0 - theta);
    }

    plt::plot3(x, y, z);
    plt::show();

}