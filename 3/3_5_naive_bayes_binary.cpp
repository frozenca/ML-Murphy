#include <cassert>
#include <iostream>
#include <initializer_list>
#include <utility>
#include <string>
#include <unordered_set>
#include <vector>
#include <variant>
#include <algorithm>

using feature_data_t = std::variant<bool, size_t, double>;

template <size_t C>
class Feature {
public:
    virtual void train(const feature_data_t& featureData, size_t classIndex) = 0;
    virtual std::vector<double> evaluate(const feature_data_t& featureData) = 0;
};

template <size_t C>
class BinaryFeature : public Feature<C> {
public:
    BinaryFeature() {
        priors.resize(C, std::pair<double, double>{1.0, 1.0});
        counts.resize(C);
        thisFeatureCounts.resize(C);
    }

    BinaryFeature(std::initializer_list<std::pair<double, double>> il) {
        assert(il.size() == C);
        for (auto& [alpha, beta] : il) {
            assert(alpha >= 1.0 && beta >= 1.0);
            priors.emplace_back(alpha, beta);
            counts.emplace_back(0);
            thisFeatureCounts.emplace_back(0);
        }
    }

    void train(const feature_data_t& featureData, size_t classIndex) final {
        assert(classIndex < C);
        bool binaryFeatureData = std::get<bool>(featureData);
        if (binaryFeatureData) {
            thisFeatureCounts[classIndex]++;
        }
        counts[classIndex]++;
    }

    std::vector<double> evaluate(const feature_data_t& featureData) final {
        bool binaryFeatureData = std::get<bool>(featureData);
        std::vector<double> probs (C);
        if (binaryFeatureData) {
            for (size_t i = 0; i < C; i++) {
                probs[i] = (priors[i].first + thisFeatureCounts[i]) / (priors[i].first + priors[i].second + counts[i]);
            }
        } else {
            for (size_t i = 0; i < C; i++) {
                probs[i] = (priors[i].second + counts[i] - thisFeatureCounts[i]) / (priors[i].first + priors[i].second + counts[i]);
            }
        }
        return probs;
    }

private:
    std::vector<std::pair<double, double>> priors;
    std::vector<size_t> thisFeatureCounts;
    std::vector<size_t> counts;
};

static const std::unordered_set<std::string> dictionary = {"secret", "offer", "low", "price", "valued",
                                                           "customer", "today", "dollar", "million", "sports",
                                                           "is", "for", "play", "healthy", "pizza"};
template <size_t C>
class NaiveBayesClassifier {
private:
    std::vector<BinaryFeature<C>> features;

public:

    NaiveBayesClassifier () {
        for (const auto& word : dictionary) {
            features.push_back(BinaryFeature<C>());
        }
    };

    std::vector<feature_data_t> preprocess(const std::string& s) {
        std::vector<feature_data_t> featureData (features.size());
        size_t featureIndex = 0;
        for (const auto& word : dictionary) {
            featureData[featureIndex] = s.find(word) != std::string::npos;
            featureIndex++;
        }
        return featureData;
    }

    void train(const std::string& s, size_t classIndex) {
        auto featureData = preprocess(s);
        for (size_t i = 0; i < features.size(); i++) {
            features[i].train(featureData[i], classIndex);
        }
    }

    size_t predict(const std::string& s) {
        auto featureData = preprocess(s);
        std::vector<std::vector<double>> probs;
        for (size_t i = 0; i < features.size(); i++) {
            probs.push_back(features[i].evaluate(featureData[i]));
        }
        std::vector<double> classProbs (C, 1.0);
        for (size_t c = 0; c < C; c++) {
            for (const auto& prob : probs) {
                classProbs[c] *= prob[c];
            }
        }
        return std::distance(classProbs.begin(), std::max_element(classProbs.begin(), classProbs.end()));
    }
};

enum class Mail {
    Spam,
    Ham
};

int main() {
    NaiveBayesClassifier<2> spamClassifier;

    spamClassifier.train("million dollar offer", static_cast<size_t>(Mail::Spam));
    spamClassifier.train("secret offer today", static_cast<size_t>(Mail::Spam));
    spamClassifier.train("secret is secret", static_cast<size_t>(Mail::Spam));

    spamClassifier.train("low price for valued customer", static_cast<size_t>(Mail::Ham));
    spamClassifier.train("play secret sports today", static_cast<size_t>(Mail::Ham));
    spamClassifier.train("sports is healthy", static_cast<size_t>(Mail::Ham));
    spamClassifier.train("low price pizza", static_cast<size_t>(Mail::Ham));

    // 0 for spam, 1 for ham
    std::cout << spamClassifier.predict("offer for million customer") << '\n';
    std::cout << spamClassifier.predict("valued healthy pizza") << '\n';
    std::cout << spamClassifier.predict("million dollar offer") << '\n';

    return 0;
}