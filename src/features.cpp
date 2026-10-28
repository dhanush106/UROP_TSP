#include "features.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace tspxgb {

double tour_cost(const Graph& graph, const std::vector<std::size_t>& tour) {
    if (tour.size() != graph.dimension) {
        throw std::runtime_error("Tour length does not match graph dimension");
    }
    double total = 0.0;
    for (std::size_t i = 0; i < tour.size(); ++i) {
        total += graph.cost(tour[i], tour[(i + 1) % tour.size()]);
    }
    return total;
}

FeatureResult compute_features(const Graph& graph,
                               std::size_t sample_factor,
                               std::uint64_t seed) {
    if (sample_factor == 0) {
        throw std::runtime_error("Sample factor must be positive");
    }
    const auto n = graph.dimension;
    if (n < 3 || sample_factor > std::numeric_limits<std::size_t>::max() / n) {
        throw std::runtime_error("Invalid graph dimension or sample factor");
    }
    const auto sample_count = sample_factor * n;
    const auto matrix_size = n * n;
    std::vector<double> sample_costs(sample_count);
    std::vector<std::vector<std::size_t>> samples(sample_count, std::vector<std::size_t>(n));
    std::vector<double> edge_frequency(matrix_size, 0.0);
    std::vector<double> edge_cost_sum(matrix_size, 0.0);
    std::vector<std::size_t> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937_64 generator(seed);
    auto best_cost = std::numeric_limits<double>::infinity();
    std::vector<std::size_t> best_tour;

    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        std::shuffle(permutation.begin(), permutation.end(), generator);
        samples[sample] = permutation;
        const auto objective = tour_cost(graph, permutation);
        sample_costs[sample] = objective;
        if (objective < best_cost) {
            best_cost = objective;
            best_tour = permutation;
        }
        for (std::size_t position = 0; position < n; ++position) {
            const auto a = permutation[position];
            const auto b = permutation[(position + 1) % n];
            const auto i = std::min(a, b);
            const auto j = std::max(a, b);
            const auto edge = i * n + j;
            edge_frequency[edge] += 1.0;
            edge_cost_sum[edge] += objective;
        }
    }

    double mean_objective = std::accumulate(sample_costs.begin(), sample_costs.end(), 0.0) /
                            static_cast<double>(sample_count);
    double objective_variance = 0.0;
    for (const auto objective : sample_costs) {
        const auto difference = objective - mean_objective;
        objective_variance += difference * difference;
    }

    auto sample_order = std::vector<std::size_t>(sample_count);
    std::iota(sample_order.begin(), sample_order.end(), 0);
    std::stable_sort(sample_order.begin(), sample_order.end(),
                     [&sample_costs](std::size_t lhs, std::size_t rhs) {
                         return sample_costs[lhs] < sample_costs[rhs];
                     });
    std::vector<std::size_t> ranks(sample_count);
    for (std::size_t rank = 0; rank < sample_count; ++rank) {
        ranks[sample_order[rank]] = rank + 1;
    }

    std::vector<double> ranking_score(matrix_size, 0.0);
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const auto weight = 1.0 / static_cast<double>(ranks[sample]);
        const auto& route = samples[sample];
        for (std::size_t position = 0; position < n; ++position) {
            const auto a = route[position];
            const auto b = route[(position + 1) % n];
            const auto i = std::min(a, b);
            const auto j = std::max(a, b);
            ranking_score[i * n + j] += weight;
        }
    }

    std::vector<double> correlation(matrix_size, 0.0);
    double minimum_correlation = std::numeric_limits<double>::infinity();
    double maximum_ranking = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const auto edge = i * n + j;
            const auto frequency = edge_frequency[edge];
            const auto variance_x = frequency * (1.0 - frequency / sample_count);
            if (variance_x > 0.0 && objective_variance > 0.0) {
                const auto covariance = edge_cost_sum[edge] - frequency * mean_objective;
                correlation[edge] = covariance / std::sqrt(variance_x * objective_variance);
            }
            minimum_correlation = std::min(minimum_correlation, correlation[edge]);
            maximum_ranking = std::max(maximum_ranking, ranking_score[edge]);
        }
    }

    std::vector<double> row_min(n, std::numeric_limits<double>::infinity());
    std::vector<double> row_max(n, -std::numeric_limits<double>::infinity());
    std::vector<double> row_mean(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            const auto value = graph.cost(i, j);
            row_min[i] = std::min(row_min[i], value);
            row_max[i] = std::max(row_max[i], value);
            row_mean[i] += value;
        }
        row_mean[i] /= static_cast<double>(n - 1);
    }

    FeatureResult result;
    result.best_sampled_tour = std::move(best_tour);
    result.best_sampled_cost = best_cost;
    result.edges.reserve(n * (n - 1) / 2);
    for (std::size_t i = 0; i < n; ++i) {
        const auto range_i = std::max(row_max[i] - row_min[i], 1.0);
        for (std::size_t j = i + 1; j < n; ++j) {
            const auto range_j = std::max(row_max[j] - row_min[j], 1.0);
            const auto value = graph.cost(i, j);
            const auto edge = i * n + j;
            const auto correlation_value =
                std::abs(minimum_correlation) > 1e-12
                    ? correlation[edge] / minimum_correlation
                    : correlation[edge];
            const auto ranking_value =
                maximum_ranking > 0.0 ? ranking_score[edge] / maximum_ranking : 0.0;
            Edge item;
            item.i = i;
            item.j = j;
            item.cost = value;
            item.features = {
                static_cast<float>(correlation_value),
                static_cast<float>(ranking_value),
                static_cast<float>((value - row_min[i]) / range_i),
                static_cast<float>((value - row_min[j]) / range_j),
                static_cast<float>((value - row_mean[i]) / range_i),
                static_cast<float>((value - row_mean[j]) / range_j),
            };
            for (const auto feature : item.features) {
                if (!std::isfinite(feature)) {
                    throw std::runtime_error("Non-finite edge feature generated");
                }
            }
            result.edges.push_back(std::move(item));
        }
    }
    return result;
}

}  // namespace tspxgb
