#include <cmath>
#include <cstddef>
#include <vector>
#include <iostream>
#include "matplotlibcpp.h"
#include <armadillo>
#include <utility>
namespace plt = matplotlibcpp;

template <typename A>
concept Arithmetic = std::is_arithmetic_v<A>;

template <typename R>
concept Floating = std::is_floating_point_v<R>;

template <typename... Args>
constexpr auto sum(Args&&... args) {
    return (args + ...);
}

template <typename... Args>
constexpr auto product(Args&&... args) {
    return (args * ...);
}

template <Floating F = double, Arithmetic A, typename... Ts>
constexpr F multivariate_beta(A a1, Ts&&... args) {
    static_assert(sizeof...(args) > 0);
    assert(a1 >= 1.0);
    F value = std::tgamma(a1) * std::tgamma(sum(args...)) / std::tgamma(a1 + sum(args...));
    if constexpr (sizeof...(args) == 1)
        return value;
    else
        return value * multivariate_beta(args...);
}

template <Floating F = double, size_t... Is, typename... Ts>
constexpr F dirichlet_impl(std::index_sequence<Is...>, Ts&&... args) {
    auto&& args_tuple = std::forward_as_tuple(std::forward<Ts>(args)...);
    if (sum(std::get<Is>(args_tuple)...) != F {1.0}) {
        return F {0.0};
    }
    constexpr size_t N = sizeof...(args) / 2;
    F beta_coeff = F {1.0} / multivariate_beta(std::get<Is + N>(args_tuple)...);
    F beta_dist = product(std::pow(std::get<Is>(args_tuple), std::get<Is + N>(args_tuple) - 1)...);
    return beta_coeff * beta_dist;
}

template <Floating F = double, typename... Ts>
constexpr F dirichlet(Ts&&... args) {
    static_assert(sizeof...(args) >= 4 && sizeof...(args) % 2 == 0);
    constexpr size_t N = sizeof...(args) / 2;
    return dirichlet_impl(std::make_index_sequence<N>(), std::forward<Ts>(args)...);
}

int main() {
    constexpr size_t N = 50;
    std::vector<std::vector<double>> x, y, z;
    double start = 0.0, end = 1.0;

    for (size_t i = 0; i < N; i++) {
        std::vector<double> x_row(N), y_row(N);
        for (size_t j = 0; j < N; j++) {
            x_row[j] = std::lerp(start, end, static_cast<double>(i) / N);
            y_row[j] = std::lerp(start, end, static_cast<double>(j) / N);
        }
        x.push_back(std::move(x_row));
        y.push_back(std::move(y_row));
    }
    for (size_t i = 0; i < N; i++) {
        std::vector<double> z_row(N);
        for (size_t j = 0; j < N; j++) {
            z_row[j] = dirichlet(x[i][j], y[i][j], 5, 5);
        }
        z.push_back(std::move(z_row));
    }
    plt::plot_surface(x, y, z);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            z[i][j] = dirichlet(x[i][j], y[i][j], 10, 10);
        }
    }
    plt::plot_surface(x, y, z);

    plt::show();
}