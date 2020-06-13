#include <cmath>
#include <cstddef>
#include <random>
#include <vector>
#include <iostream>
#include <numeric>
#include "matplotlibcpp.h"
#include <armadillo>
namespace plt = matplotlibcpp;

int main() {

    std::mt19937 gen(std::random_device{}());
    // N = 10, p = 0.7
    constexpr size_t N = 10;
    constexpr double p = 0.7;
    std::binomial_distribution<> binom(N, p);

    size_t SAMPLES = 100;
    size_t NUM_RVS = 100;

    std::vector<size_t> x(N);
    std::vector<size_t> y(N);
    std::iota(x.begin(), x.end(), 0);

    for (size_t i = 0; i < SAMPLES; i++) {
        size_t value = 0;
        for (size_t j = 0; j < NUM_RVS; j++) {
            value += binom(gen);
        }
        value /= NUM_RVS;
        y[value]++;
    }
    plt::plot(x, y, "bo");


    plt::show();
}
