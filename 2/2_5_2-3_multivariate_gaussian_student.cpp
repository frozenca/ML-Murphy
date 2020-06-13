#include <cmath>
#include <vector>
#include "matplotlibcpp.h"
#include <armadillo>
namespace plt = matplotlibcpp;

template <typename N>
concept Unsigned = std::is_unsigned_v<N>;

template <typename R>
concept Floating = std::is_floating_point_v<R>;

template <size_t D>
double gaussian_pdf(arma::vec& x, arma::vec& mu, arma::mat& Sigma) {
    auto diff = x - mu;
    double dist = -0.5 * dot(diff.t(), inv(Sigma) * diff);
    double det_value = det(Sigma);
    return std::exp(dist) / (std::pow(2.0 * M_PI, D / 2.0) * std::sqrt(det_value));
}

template <size_t D>
double student_pdf(arma::vec& x, arma::vec& mu, arma::mat& Sigma, double nu) {
    auto diff = x - mu;
    auto V = nu * Sigma;
    double normalizer = std::tgamma((nu + D) / 2.0) / (std::tgamma(nu / 2.0) * std::sqrt(det(M_PI * V)));
    double dist = 1 + dot(diff.t(), inv(V) * diff);
    double prob = std::pow(dist, -0.5 * (nu + D));
    return prob * normalizer;
}

int main() {
    constexpr size_t N = 50;
    std::vector<std::vector<double>> x, y, z;
    double start = -5.0, end = 5.0;

    for (size_t i = 0; i < N; i++) {
        std::vector<double> x_row(N), y_row(N);
        for (size_t j = 0; j < N; j++) {
            x_row[j] = std::lerp(start, end, static_cast<double>(i) / N);
            y_row[j] = std::lerp(start, end, static_cast<double>(j) / N);
        }
        x.push_back(std::move(x_row));
        y.push_back(std::move(y_row));
    }

    arma::vec mu = arma::zeros<arma::vec>(2);
    arma::mat Sigma = arma::eye<arma::mat>(2, 2);
    for (size_t i = 0; i < N; i++) {
        std::vector<double> z_row(N);
        for (size_t j = 0; j < N; j++) {
            arma::vec v {x[i][j], y[i][j]};
            z_row[j] = gaussian_pdf<2>(v, mu, Sigma);
        }
        z.push_back(std::move(z_row));
    }
    plt::plot_surface(x, y, z);

    mu = {1.0, -2.0};
    Sigma = {{3, 0.4}, {0.4, 0.5}};
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            arma::vec v {x[i][j], y[i][j]};
            z[i][j] = gaussian_pdf<2>(v, mu, Sigma);
        }
    }
    plt::plot_surface(x, y, z);

    mu = arma::zeros<arma::vec>(2);
    Sigma = arma::eye<arma::mat>(2, 2);
    double nu = 4.0;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            arma::vec v {x[i][j], y[i][j]};
            z[i][j] = student_pdf<2>(v, mu, Sigma, nu);
        }
    }
    plt::plot_surface(x, y, z);

    mu = {1.0, -2.0};
    Sigma = {{3, 0.4}, {0.4, 0.5}};
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            arma::vec v {x[i][j], y[i][j]};
            z[i][j] = student_pdf<2>(v, mu, Sigma, nu);
        }
    }
    plt::plot_surface(x, y, z);
    plt::show();
}