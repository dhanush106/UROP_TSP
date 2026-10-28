#ifndef TSP_XGBOOST_FEATURES_H
#define TSP_XGBOOST_FEATURES_H

#include "graph.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace tspxgb {

constexpr std::size_t kFeatureCount = 6;

struct Edge {
    std::size_t i = 0;
    std::size_t j = 0;
    double cost = 0.0;
    std::uint8_t label = 0;
    std::vector<float> features;
    float score = 0.0F;
    bool predicted_keep = false;
    bool keep = false;
};

struct FeatureResult {
    std::vector<Edge> edges;
    std::vector<std::size_t> best_sampled_tour;
    double best_sampled_cost = 0.0;
};

FeatureResult compute_features(const Graph& graph,
                               std::size_t sample_factor,
                               std::uint64_t seed);
double tour_cost(const Graph& graph, const std::vector<std::size_t>& tour);

}  // namespace tspxgb

#endif
