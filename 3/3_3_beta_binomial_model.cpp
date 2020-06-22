#include <random>
#include <cmath>
#include <iostream>
#include <cassert>

std::mt19937 gen(std::random_device{}());
std::uniform_real_distribution<> theta_dist(0, 1);

class BetaBinomial {
public:
    BetaBinomial() {
        true_theta = theta_dist(gen);
    }

    BetaBinomial(double a, double b) : a {a}, b {b} {
        assert(a >= 1 && b >= 1);
        true_theta = theta_dist(gen);
    }

    size_t draw(size_t count = 1) {
        std::binomial_distribution<> d(count, true_theta);
        size_t h = d(gen);
        head += h;
        tail += count - h;
        update_posterior();
        std::cout << "After drawing " << count << " times from Beta-Binomial distribution with "
                  << "prior (" << a << ", " << b << "), we get "
                  << h << " heads and " << (count - h) << " tails.\n"
                  << "MLE mean : " << theta_MLE << ", MLE variance : " << sigma_MLE << '\n'
                << "MAP mean : " << theta_MAP << ", MAP variance : " << sigma_MAP << '\n'
                << "Posterior mean : " << theta_bar << '\n';
        return h;
    }

    size_t draw_from_posterior(size_t count = 1) {
        std::binomial_distribution<> d(count, theta_MAP);
        size_t h = d(gen);
        return h;
    }

    [[nodiscard]] double getA() { return a; }
    [[nodiscard]] double getB() { return b; }
    [[nodiscard]] double getMLEMean() { return theta_MLE; }
    [[nodiscard]] double getMAP() { return theta_MAP; }
    [[nodiscard]] double getPosteriorMean() { return theta_bar; }
    [[nodiscard]] double getMLEVariance() { return sigma_MLE; }
    [[nodiscard]] double getPosteriorVariance() { return sigma_MAP; }

private:
    void update_posterior() {
        theta_MAP = (a + head - 1) / static_cast<double>(a + b + head + tail - 2);
        theta_MLE = head / static_cast<double>(head + tail);
        theta_bar = (a + head) / static_cast<double>(a + b + head + tail);
        sigma_MLE = (1.0 + head) * (1.0 + tail) /
                    (static_cast<double>(std::pow(2.0 + head + tail, 2)) *
                     (3.0 + head + tail));
        sigma_MAP = (a + head) * (b + tail) /
                (static_cast<double>(std::pow(a + b + head + tail, 2)) *
                        (a + b + head + tail + 1));
    }

    size_t head = 0; // empirical counts
    size_t tail = 0;
    double true_theta;
    double a = 1; // prior strength
    double b = 1;
    double theta_MLE;
    double theta_MAP;
    double theta_bar;
    double sigma_MLE;
    double sigma_MAP;
};

int main() {
    BetaBinomial betabinom; // beta-binomial with unknown parameter & uniform prior
    betabinom.draw(10);
    betabinom.draw(100);
    betabinom.draw(1000);

    betabinom = BetaBinomial(50, 50); // beta-binomial with weighted, even prior
    betabinom.draw(10);
    betabinom.draw(100);
    betabinom.draw(1000);

    betabinom = BetaBinomial(1, 100); // beta-binomial with skewed prior
    betabinom.draw(10);
    betabinom.draw(100);
    betabinom.draw(1000);
}