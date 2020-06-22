#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <array>
#include <cassert>

std::mt19937 gen(std::random_device{}());
std::uniform_real_distribution<> theta_dist(0, 1);

template <size_t N = 3>
class DirichletMultinomial {
public:
    DirichletMultinomial() {
        static_assert(N >= 2);
        std::fill(alpha.begin(), alpha.end(), 1.0);
        std::fill(empirical_counts.begin(), empirical_counts.end(), 0);
        alpha_zero = N;
        sample_from_simplex();
    }

    DirichletMultinomial(const std::array<double, N>& alpha) : alpha {alpha} {
        static_assert(N >= 2);
        assert(std::all_of(alpha.begin(), alpha.end(), [](double d){ return d >= 1.0;}));
        std::fill(empirical_counts.begin(), empirical_counts.end(), 0);
        alpha_zero = std::accumulate(alpha.begin(), alpha.end(), 0.0);
        sample_from_simplex();
    }

    void draw(size_t count = 1) {
        for (size_t i = 0; i < count; i++) {
            empirical_counts[sample()]++;
            total_counts++;
        }
        update_posterior();
        std::cout << "After drawing " << count << " times from Beta-Binomial distribution with "
                  << "prior (";
        for (size_t i = 0; i < N - 1; i++) {
            std::cout << alpha[i] << ", ";
        }
        std::cout << alpha[N - 1] << "),\n we get (";
        for (size_t i = 0; i < N - 1; i++) {
            std::cout << empirical_counts[i] << ", ";
        }
        std::cout << empirical_counts[N - 1] << ") counts.\n";
        std::cout << "MLE mean : (";
        for (size_t i = 0; i < N - 1; i++) {
            std::cout << theta_MLE[i] << ", ";
        }
        std::cout << theta_MLE[N - 1] << "),\n";
        std::cout << "MAP mean : (";
        for (size_t i = 0; i < N - 1; i++) {
            std::cout << theta_MAP[i] << ", ";
        }
        std::cout << theta_MAP[N - 1] << ").\n";
    }

private:
    void update_posterior() {
        for (size_t i = 0; i < N; i++) {
            posterior_strength[i] = alpha[i] + empirical_counts[i];
            theta_MLE[i] = empirical_counts[i] / static_cast<double>(total_counts);
            theta_MAP[i] = (empirical_counts[i] + alpha[i] - 1) / static_cast<double>(total_counts + alpha_zero - N);
        }
    }

    void sample_from_simplex() {
        std::array<double, N + 1> x = {0.0};
        x[0] = 0.0;
        for (size_t i = 1; i < N; i++) {
            x[i] = theta_dist(gen);
        }
        x[N] = 1.0;
        std::sort(x.begin(), x.end());
        for (size_t i = 0; i < N; i++) {
            true_theta[i] = x[i + 1] - x[i];
        }
    }

    size_t sample() {
        double value = theta_dist(gen);
        size_t i = 0;
        while (value > 0) {
            value -= true_theta[i++];
        }
        return i - 1;
    }

    std::array<size_t, N> empirical_counts;
    size_t total_counts = 0;
    std::array<double, N> alpha; // prior strength
    double alpha_zero = 0;
    std::array<double, N> true_theta;
    std::array<double, N> posterior_strength;
    std::array<double, N> theta_MLE;
    std::array<double, N> theta_MAP;
};

int main() {
    DirichletMultinomial multinomial;
    multinomial.draw(10);
    multinomial.draw(100);
    multinomial.draw(1000);

    multinomial = DirichletMultinomial<3>(std::array<double, 3>({30.0, 30.0, 30.0}));
    multinomial.draw(10);
    multinomial.draw(100);
    multinomial.draw(1000);

    multinomial = DirichletMultinomial<3>(std::array<double, 3>({1.0, 10.0, 100.0}));
    multinomial.draw(10);
    multinomial.draw(100);
    multinomial.draw(1000);
}