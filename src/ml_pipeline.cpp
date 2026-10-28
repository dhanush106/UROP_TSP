#include "ml_pipeline.h"

#include "features.h"
#include "graph.h"
#include "solver.h"
#include "xgboost_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace tspxgb {
namespace {

struct MatrixGuard {
    const XGBoostApi& api;
    DMatrixHandle handle;
    ~MatrixGuard() {
        if (handle != nullptr) {
            api.free_matrix(handle);
        }
    }
};

struct BoosterGuard {
    const XGBoostApi& api;
    BoosterHandle handle;
    ~BoosterGuard() {
        if (handle != nullptr) {
            api.free_booster(handle);
        }
    }
};

struct Dataset {
    std::vector<float> features;
    std::vector<float> labels;
    std::vector<std::size_t> groups;
    std::vector<std::string> instances;
    std::vector<std::size_t> city_i;
    std::vector<std::size_t> city_j;
    std::vector<double> edge_costs;

    std::size_t rows() const {
        return labels.size();
    }
};

struct MetricSummary {
    double recall = 0.0;
    double precision = 0.0;
    double retained_fraction = 0.0;
    double complete_tour_fraction = 0.0;
    std::size_t true_positive = 0;
    std::size_t false_positive = 0;
    std::size_t false_negative = 0;
    std::size_t true_negative = 0;
};

std::string join_path(const std::string& directory, const std::string& filename) {
    if (directory.empty()) {
        return filename;
    }
    const auto last = directory.back();
    return directory + ((last == '/' || last == '\\') ? "" :
#ifdef _WIN32
                        "\\"
#else
                        "/"
#endif
                        ) + filename;
}

void make_directory(const std::string& path) {
#ifdef _WIN32
    if (CreateDirectoryA(path.c_str(), nullptr) == 0 &&
        GetLastError() != ERROR_ALREADY_EXISTS) {
        throw std::runtime_error("Cannot create directory: " + path);
    }
#else
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + path);
    }
#endif
}

void ensure_directory(const std::string& path) {
    if (path.empty()) {
        return;
    }
    std::string current;
    std::size_t position = 0;
#ifdef _WIN32
    if (path.size() > 2 && path[1] == ':') {
        current = path.substr(0, 2);
        position = 2;
    }
#endif
    while (position < path.size()) {
        while (position < path.size() && (path[position] == '/' || path[position] == '\\')) {
            current.push_back(path[position++]);
        }
        const auto start = position;
        while (position < path.size() && path[position] != '/' && path[position] != '\\') {
            ++position;
        }
        if (position > start) {
            current.append(path, start, position - start);
            make_directory(current);
        }
    }
}

void ensure_parent_directory(const std::string& file_path) {
    const auto separator = file_path.find_last_of("/\\");
    if (separator != std::string::npos) {
        ensure_directory(file_path.substr(0, separator));
    }
}

std::string basename_stem(const std::string& path) {
    const auto separator = path.find_last_of("/\\");
    const auto filename = separator == std::string::npos ? path : path.substr(separator + 1);
    const auto extension = filename.find_last_of('.');
    return extension == std::string::npos ? filename : filename.substr(0, extension);
}

std::string find_tour_file(const std::string& tsp_path) {
    const auto extension = tsp_path.find_last_of('.');
    const auto stem = extension == std::string::npos ? tsp_path : tsp_path.substr(0, extension);
    for (const auto& extension : {".sol", ".opt.tour", ".tour"}) {
        const auto candidate = stem + extension;
        std::ifstream input(candidate.c_str());
        if (input.good()) {
            return candidate;
        }
    }
    throw std::runtime_error("No .sol, .opt.tour, or .tour file for " + tsp_path);
}

std::vector<std::string> list_tsp_files(const std::string& directory) {
    std::vector<std::string> paths;
#ifdef _WIN32
    WIN32_FIND_DATAA entry;
    const auto search = join_path(directory, "*.tsp");
    const auto handle = FindFirstFileA(search.c_str(), &entry);
    if (handle == INVALID_HANDLE_VALUE) {
        return paths;
    }
    do {
        if ((entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
            paths.push_back(join_path(directory, entry.cFileName));
        }
    } while (FindNextFileA(handle, &entry) != 0);
    FindClose(handle);
#else
    DIR* handle = opendir(directory.c_str());
    if (handle == nullptr) {
        return paths;
    }
    while (const auto* entry = readdir(handle)) {
        const std::string filename = entry->d_name;
        if (filename.size() > 4 &&
            filename.substr(filename.size() - 4) == ".tsp") {
            paths.push_back(join_path(directory, filename));
        }
    }
    closedir(handle);
#endif
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::string get_library_path(const std::string& requested) {
    if (!requested.empty()) {
        return requested;
    }
    if (const auto* environment = std::getenv("XGBOOST_LIB")) {
        if (*environment != '\0') {
            return environment;
        }
    }
    throw std::runtime_error(
        "Set XGBOOST_LIB to the XGBoost shared library path or pass --xgboost-lib");
}

void add_instance(Dataset& dataset,
                  const Graph& graph,
                  const FeatureResult& features,
                  const std::vector<std::size_t>& tour,
                  std::size_t group,
                  const std::string& instance_name) {
    const auto n = graph.dimension;
    std::vector<bool> positive(n * n, false);
    for (std::size_t k = 0; k < n; ++k) {
        const auto a = tour[k];
        const auto b = tour[(k + 1) % n];
        positive[std::min(a, b) * n + std::max(a, b)] = true;
    }
    for (const auto& edge : features.edges) {
        dataset.features.insert(dataset.features.end(), edge.features.begin(),
                                edge.features.end());
        dataset.labels.push_back(positive[edge.i * n + edge.j] ? 1.0F : 0.0F);
        dataset.groups.push_back(group);
        dataset.instances.push_back(instance_name);
        dataset.city_i.push_back(edge.i + 1);
        dataset.city_j.push_back(edge.j + 1);
        dataset.edge_costs.push_back(edge.cost);
    }
}

std::vector<std::size_t> split_group_ids(std::size_t count, std::uint64_t seed) {
    if (count < 4) {
        throw std::runtime_error("At least four training instances are required");
    }
    std::vector<std::size_t> groups(count);
    std::iota(groups.begin(), groups.end(), 0);
    std::mt19937_64 generator(seed);
    std::shuffle(groups.begin(), groups.end(), generator);
    return groups;
}

double average_precision(const std::vector<float>& scores,
                         const std::vector<float>& labels) {
    std::vector<std::size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&scores](std::size_t lhs, std::size_t rhs) {
                         return scores[lhs] > scores[rhs];
                     });
    const auto positives = static_cast<std::size_t>(
        std::count(labels.begin(), labels.end(), 1.0F));
    if (positives == 0) {
        return 0.0;
    }
    std::size_t found = 0;
    double precision_sum = 0.0;
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
        if (labels[order[rank]] == 1.0F) {
            ++found;
            precision_sum += static_cast<double>(found) / (rank + 1);
        }
    }
    return precision_sum / positives;
}

double choose_threshold(const std::vector<float>& scores,
                       const std::vector<float>& labels,
                       double recall_target) {
    std::vector<float> positive_scores;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == 1.0F) {
            positive_scores.push_back(scores[i]);
        }
    }
    if (positive_scores.empty()) {
        throw std::runtime_error("Validation data has no positive edges");
    }
    std::sort(positive_scores.begin(), positive_scores.end());
    const auto allowed_false_negatives = static_cast<std::size_t>(
        std::floor((1.0 - recall_target) * positive_scores.size()));
    return positive_scores[std::min(allowed_false_negatives,
                                    positive_scores.size() - 1)];
}

MetricSummary evaluate(const Dataset& dataset,
                       const std::vector<float>& scores,
                       double threshold) {
    if (dataset.rows() != scores.size()) {
        throw std::runtime_error("Prediction count does not match dataset row count");
    }
    MetricSummary summary;
    std::map<std::size_t, std::pair<std::size_t, std::size_t>> tour_edges;
    std::size_t retained = 0;
    for (std::size_t i = 0; i < dataset.rows(); ++i) {
        const bool actual = dataset.labels[i] == 1.0F;
        const bool predicted = scores[i] >= threshold;
        if (predicted) {
            ++retained;
        }
        if (predicted && actual) {
            ++summary.true_positive;
        } else if (predicted) {
            ++summary.false_positive;
        } else if (actual) {
            ++summary.false_negative;
        } else {
            ++summary.true_negative;
        }
        if (actual) {
            ++tour_edges[dataset.groups[i]].second;
            if (predicted) {
                ++tour_edges[dataset.groups[i]].first;
            }
        }
    }
    const auto positives = summary.true_positive + summary.false_negative;
    const auto predicted_positive = summary.true_positive + summary.false_positive;
    summary.recall = positives == 0
                         ? 0.0
                         : static_cast<double>(summary.true_positive) / positives;
    summary.precision = predicted_positive == 0
                            ? 0.0
                            : static_cast<double>(summary.true_positive) /
                                  predicted_positive;
    summary.retained_fraction =
        dataset.rows() == 0 ? 0.0 : static_cast<double>(retained) / dataset.rows();
    if (!tour_edges.empty()) {
        const auto complete = std::count_if(
            tour_edges.begin(), tour_edges.end(),
            [](const auto& item) { return item.second.first == item.second.second; });
        summary.complete_tour_fraction =
            static_cast<double>(complete) / tour_edges.size();
    }
    return summary;
}

void write_dataset_splits(const std::string& directory,
                          const std::array<Dataset, 3>& splits,
                          const std::vector<std::vector<std::string>>& split_names) {
    ensure_directory(directory);
    const std::array<const char*, 3> names = {
        "train", "validation", "test"
    };
    std::ofstream assignments(join_path(directory, "instance_splits.csv").c_str());
    std::ofstream summary(join_path(directory, "dataset_summary.csv").c_str());
    if (!assignments || !summary) {
        throw std::runtime_error("Cannot write dataset split metadata in " + directory);
    }
    assignments << "instance,split\n";
    summary << "split,instance_count,edge_rows,positive_edges,negative_edges\n";
    for (std::size_t split = 0; split < splits.size(); ++split) {
        for (const auto& instance : split_names[split]) {
            assignments << instance << ',' << names[split] << '\n';
        }
        const auto positives = static_cast<std::size_t>(
            std::count(splits[split].labels.begin(), splits[split].labels.end(), 1.0F));
        summary << names[split] << ',' << split_names[split].size() << ','
                << splits[split].rows() << ',' << positives << ','
                << splits[split].rows() - positives << '\n';

        std::ofstream data(join_path(directory,
                                     std::string(names[split]) + "_data.csv").c_str());
        if (!data) {
            throw std::runtime_error("Cannot write " + std::string(names[split]) +
                                     " edge data in " + directory);
        }
        data << "instance,city_i,city_j,edge_cost,label,correlation_normalized,"
                "ranking_normalized,cost_vs_min_i,cost_vs_min_j,"
                "cost_vs_mean_i,cost_vs_mean_j\n";
        data << std::setprecision(9);
        for (std::size_t row = 0; row < splits[split].rows(); ++row) {
            data << splits[split].instances[row] << ','
                 << splits[split].city_i[row] << ','
                 << splits[split].city_j[row] << ','
                 << splits[split].edge_costs[row] << ','
                 << splits[split].labels[row];
            const auto offset = row * kFeatureCount;
            for (std::size_t feature = 0; feature < kFeatureCount; ++feature) {
                data << ',' << splits[split].features[offset + feature];
            }
            data << '\n';
        }
    }
}

void write_edge_metrics(const std::string& directory,
                        const std::array<MetricSummary, 3>& metrics,
                        const std::array<double, 3>& average_precisions) {
    std::ofstream output(join_path(directory, "edge_metrics.csv").c_str());
    if (!output) {
        throw std::runtime_error("Cannot write edge metrics in " + directory);
    }
    output << "split,average_precision,edge_recall,edge_precision,"
              "retained_edge_fraction,complete_tour_fraction,"
              "true_positive,false_positive,false_negative,true_negative\n";
    const std::array<const char*, 3> names = {
        "train", "validation", "test"
    };
    output << std::setprecision(10);
    for (std::size_t split = 0; split < metrics.size(); ++split) {
        const auto& item = metrics[split];
        output << names[split] << ',' << average_precisions[split] << ','
               << item.recall << ',' << item.precision << ','
               << item.retained_fraction << ',' << item.complete_tour_fraction
               << ',' << item.true_positive << ',' << item.false_positive
               << ',' << item.false_negative << ',' << item.true_negative << '\n';
    }
}

void write_metadata(const std::string& path,
                    double threshold,
                    std::size_t best_iteration,
                    double best_ap,
                    double recall_target,
                    std::uint64_t seed,
                    std::size_t sample_factor,
                    std::size_t max_rounds,
                    std::size_t early_stopping_rounds,
                    std::size_t threads,
                    double scale_pos_weight) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot write model metadata: " + path);
    }
    output << std::setprecision(17)
           << "format_version=1\n"
           << "threshold=" << threshold << "\n"
           << "best_iteration=" << best_iteration << "\n"
           << "validation_average_precision=" << best_ap << "\n"
           << "recall_target=" << recall_target << "\n"
           << "seed=" << seed << "\n"
           << "sample_factor=" << sample_factor << "\n"
           << "max_rounds=" << max_rounds << "\n"
           << "early_stopping_rounds=" << early_stopping_rounds << "\n"
           << "xgboost_threads=" << threads << "\n"
           << "scale_pos_weight=" << scale_pos_weight << "\n"
           << "objective=binary:logistic\n"
           << "eval_metric=aucpr\n"
           << "tree_method=hist\n"
           << "max_depth=5\n"
           << "eta=0.05\n"
           << "subsample=0.8\n"
           << "colsample_bytree=0.9\n"
           << "min_child_weight=5\n"
           << "lambda=1.0\n"
           << "features=correlation_normalized,ranking_normalized,cost_vs_min_i,"
              "cost_vs_min_j,cost_vs_mean_i,cost_vs_mean_j\n";
}

struct ModelMetadata {
    double threshold = 0.5;
    std::size_t best_iteration = 0;
};

ModelMetadata read_metadata(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open model metadata: " + path);
    }
    ModelMetadata metadata;
    std::string line;
    bool found_threshold = false;
    bool found_iteration = false;
    while (std::getline(input, line)) {
        const auto delimiter = line.find('=');
        if (delimiter == std::string::npos) {
            continue;
        }
        const auto key = line.substr(0, delimiter);
        const auto value = line.substr(delimiter + 1);
        if (key == "threshold") {
            metadata.threshold = std::stod(value);
            found_threshold = true;
        } else if (key == "best_iteration") {
            metadata.best_iteration = static_cast<std::size_t>(std::stoul(value));
            found_iteration = true;
        }
    }
    if (!found_threshold || !found_iteration ||
        !std::isfinite(metadata.threshold) ||
        metadata.threshold < 0.0 || metadata.threshold > 1.0) {
        throw std::runtime_error("Invalid model metadata: " + path);
    }
    return metadata;
}

void write_results(const std::string& output_root,
                   const std::string& instance_name,
                   const Graph& graph,
                   const FeatureResult& feature_result,
                   const std::vector<Edge>& edges,
                   double threshold,
                   double read_seconds,
                   double sampling_seconds,
                   double reduction_seconds,
                   double total_seconds,
                   const SolverResult& solver) {
    ensure_directory(output_root);
    const auto directory = join_path(output_root, instance_name);
    ensure_directory(directory);
    std::ofstream all_edges(join_path(directory, "edge_predictions.csv").c_str());
    std::ofstream reduced_edges(join_path(directory, "reduced_edges.csv").c_str());
    std::ofstream tour_file(join_path(directory, "best_sampled_tour.txt").c_str());
    std::ofstream solver_tour(join_path(output_root, instance_name + "_res_sol.csv").c_str());
    std::ofstream objective(join_path(output_root, instance_name + "_res_obj.csv").c_str());
    std::ofstream timing(join_path(output_root, instance_name + "_res_time.csv").c_str());
    std::ofstream summary(join_path(directory, "summary.txt").c_str());
    if (!all_edges || !reduced_edges || !tour_file || !summary) {
        throw std::runtime_error("Cannot create reduction output files in " + directory);
    }
    if (!solver_tour || !objective || !timing) {
        throw std::runtime_error("Cannot create benchmark CSV files in " + output_root);
    }
    all_edges << "city_i,city_j,cost,score,predicted_keep,kept_after_repair,"
                 "correlation_normalized,ranking_normalized,cost_vs_min_i,"
                 "cost_vs_min_j,cost_vs_mean_i,cost_vs_mean_j\n";
    reduced_edges << "city_i,city_j,cost,score\n";
    std::size_t kept_count = 0;
    std::size_t repaired = 0;
    for (const auto& edge : edges) {
        all_edges << edge.i + 1 << ',' << edge.j + 1 << ',' << edge.cost << ','
                  << edge.score << ',' << edge.predicted_keep << ',' << edge.keep;
        for (const auto feature : edge.features) {
            all_edges << ',' << feature;
        }
        all_edges << '\n';
        if (edge.keep) {
            ++kept_count;
            reduced_edges << edge.i + 1 << ',' << edge.j + 1 << ',' << edge.cost
                          << ',' << edge.score << '\n';
            if (!edge.predicted_keep) {
                ++repaired;
            }
        }
    }
    for (const auto city : feature_result.best_sampled_tour) {
        tour_file << city + 1 << '\n';
    }
    const auto total = edges.size();
    const auto retained_directed = kept_count * 2;
    solver_tour << "TOUR_SECTION\n";
    if (solver.has_solution) {
        for (const auto city : solver.tour) {
            solver_tour << city + 1 << '\n';
        }
        solver_tour << "EOF\n";
    }
    objective << "original_directed_arcs,sampled_tour_objective,"
                 "reduced_directed_arcs,solver_objective,best_bound,"
                 "relative_gap,solver_status,is_optimal\n";
    objective << std::setprecision(12) << total * 2 << ','
              << feature_result.best_sampled_cost << ',' << retained_directed << ',';
    if (solver.has_solution) {
        objective << solver.objective << ',';
        if (std::isfinite(solver.best_bound)) {
            objective << solver.best_bound;
        }
        objective << ',';
        if (std::isfinite(solver.relative_gap)) {
            objective << solver.relative_gap;
        }
    } else {
        objective << ",,";
    }
    objective << ',' << solver.status << ',' << (solver.optimal ? 1 : 0) << '\n';
    timing << "read_seconds,sampling_seconds,reduction_seconds,solver_seconds,"
              "total_seconds\n"
           << std::setprecision(12) << read_seconds << ',' << sampling_seconds
           << ',' << reduction_seconds << ',' << solver.elapsed_seconds << ','
           << total_seconds << '\n';
    summary << std::setprecision(10)
            << "instance=" << graph.name << '\n'
            << "threshold=" << threshold << '\n'
            << "sampled_tour_cost=" << feature_result.best_sampled_cost << '\n'
            << "original_undirected_edges=" << total << '\n'
            << "retained_undirected_edges=" << kept_count << '\n'
            << "retained_edge_fraction="
            << (total == 0 ? 0.0 : static_cast<double>(kept_count) / total) << '\n'
            << "removed_edge_fraction="
            << (total == 0 ? 0.0 : 1.0 - static_cast<double>(kept_count) / total) << '\n'
            << "repaired_tour_edges=" << repaired << '\n'
            << "all_sampled_tour_edges_retained=true\n"
            << "read_seconds=" << read_seconds << '\n'
            << "sampling_seconds=" << sampling_seconds << '\n'
            << "reduction_seconds=" << reduction_seconds << '\n'
            << "solver_status=" << solver.status << '\n'
            << "solver_has_solution=" << (solver.has_solution ? "true" : "false") << '\n'
            << "solver_optimal=" << (solver.optimal ? "true" : "false") << '\n'
            << "solver_objective=";
    if (solver.has_solution) {
        summary << solver.objective << '\n' << "solver_best_bound=";
        if (std::isfinite(solver.best_bound)) {
            summary << solver.best_bound;
        } else {
            summary << "not_available";
        }
        summary << '\n' << "solver_relative_gap=";
        if (std::isfinite(solver.relative_gap)) {
            summary << solver.relative_gap;
        } else {
            summary << "not_available";
        }
        summary << '\n';
    } else {
        summary << "not_available\n"
                << "solver_best_bound=not_available\n"
                << "solver_relative_gap=not_available\n";
    }
    summary << "solver_seconds=" << solver.elapsed_seconds << '\n'
            << "total_seconds=" << total_seconds << '\n';
}

}  // namespace

void train_model(const TrainOptions& options) {
    if (options.recall_target <= 0.0 || options.recall_target > 1.0 ||
        options.rounds == 0 || options.early_stopping_rounds == 0) {
        throw std::runtime_error("Invalid training rounds or recall target");
    }
    const auto instances = list_tsp_files(options.data_directory);
    if (instances.size() < 4) {
        throw std::runtime_error("At least four .tsp training instances are required");
    }
    const auto shuffled = split_group_ids(instances.size(), options.seed);
    const auto train_end = std::max<std::size_t>(1, instances.size() * 70 / 100);
    const auto validation_end = std::max(train_end + 1, instances.size() * 85 / 100);
    std::vector<std::size_t> assignment(instances.size(), 2);
    for (std::size_t index = 0; index < train_end; ++index) {
        assignment[shuffled[index]] = 0;
    }
    for (std::size_t index = train_end; index < validation_end; ++index) {
        assignment[shuffled[index]] = 1;
    }

    std::array<Dataset, 3> split_data;
    std::vector<std::vector<std::string>> split_names(3);
    std::mt19937_64 sampling_generator(options.seed);
    for (std::size_t index = 0; index < instances.size(); ++index) {
        const auto& instance_path = instances[index];
        const auto graph = Graph::read_tsplib(instance_path);
        const auto tour = graph.read_tour(find_tour_file(instance_path));
        auto features = compute_features(graph, options.sample_factor, sampling_generator());
        const auto split = assignment[index];
        const auto instance_name = basename_stem(instance_path);
        add_instance(split_data[split], graph, features, tour, index, instance_name);
        split_names[split].push_back(instance_name);
        std::cout << '[' << index + 1 << '/' << instances.size() << "] "
                  << basename_stem(instance_path) << "  edges="
                  << features.edges.size() << "  split="
                  << (split == 0 ? "train" : split == 1 ? "validation" : "test")
                  << std::endl;
    }
    if (split_data[0].rows() == 0 || split_data[1].rows() == 0 ||
        split_data[2].rows() == 0) {
        throw std::runtime_error("A training/validation/test split is empty");
    }
    write_dataset_splits(options.dataset_output_directory, split_data, split_names);
    const auto positive_count = static_cast<double>(
        std::count(split_data[0].labels.begin(), split_data[0].labels.end(), 1.0F));
    const auto negative_count = static_cast<double>(split_data[0].rows()) - positive_count;
    if (positive_count == 0.0) {
        throw std::runtime_error("Training split contains no positive examples");
    }
    const auto scale_pos_weight = negative_count / positive_count;

    XGBoostApi api(get_library_path(options.xgboost_library));
    MatrixGuard training_matrix{api, api.create_matrix(
        split_data[0].features, split_data[0].rows(), kFeatureCount)};
    api.set_labels(training_matrix.handle, split_data[0].labels);
    MatrixGuard validation_matrix{api, api.create_matrix(
        split_data[1].features, split_data[1].rows(), kFeatureCount)};
    api.set_labels(validation_matrix.handle, split_data[1].labels);

    BoosterGuard booster{api, api.create_booster(training_matrix.handle)};
    api.set_parameter(booster.handle, "objective", "binary:logistic");
    api.set_parameter(booster.handle, "eval_metric", "aucpr");
    api.set_parameter(booster.handle, "tree_method", "hist");
    api.set_parameter(booster.handle, "max_depth", "5");
    api.set_parameter(booster.handle, "eta", "0.05");
    api.set_parameter(booster.handle, "subsample", "0.8");
    api.set_parameter(booster.handle, "colsample_bytree", "0.9");
    api.set_parameter(booster.handle, "min_child_weight", "5");
    api.set_parameter(booster.handle, "lambda", "1.0");
    api.set_parameter(booster.handle, "scale_pos_weight",
                      std::to_string(scale_pos_weight));
    api.set_parameter(booster.handle, "seed", std::to_string(options.seed));
    api.set_parameter(booster.handle, "nthread", std::to_string(options.threads));

    ensure_parent_directory(options.model_path);
    std::ofstream training_history(
        join_path(options.dataset_output_directory, "training_history.csv").c_str());
    if (!training_history) {
        throw std::runtime_error("Cannot write training history in " +
                                 options.dataset_output_directory);
    }
    training_history << "round,validation_average_precision,"
                        "best_validation_average_precision,"
                        "best_iteration_zero_based,is_best\n";
    training_history << std::setprecision(10);
    auto best_average_precision = -1.0;
    std::size_t best_iteration = 0;
    std::size_t rounds_without_improvement = 0;
    for (std::size_t iteration = 0; iteration < options.rounds; ++iteration) {
        api.update(booster.handle, static_cast<int>(iteration), training_matrix.handle);
        const auto scores = api.predict(booster.handle, validation_matrix.handle);
        const auto ap = average_precision(scores, split_data[1].labels);
        const bool improved = ap > best_average_precision + 1e-12;
        if (improved) {
            best_average_precision = ap;
            best_iteration = iteration;
            rounds_without_improvement = 0;
            api.save_model(booster.handle, options.model_path);
        } else {
            ++rounds_without_improvement;
        }
        training_history << iteration + 1 << ',' << ap << ','
                         << best_average_precision << ',' << best_iteration << ','
                         << (improved ? 1 : 0) << '\n';
        training_history.flush();
        if (!training_history) {
            throw std::runtime_error("Failed while writing training history");
        }
        if ((iteration + 1) % 25 == 0 || iteration == 0) {
            std::cout << "round=" << iteration + 1 << " validation_ap=" << ap
                      << " best_ap=" << best_average_precision << std::endl;
        }
        if (rounds_without_improvement >= options.early_stopping_rounds) {
            break;
        }
    }
    if (best_average_precision < 0.0) {
        throw std::runtime_error("XGBoost training produced no valid model");
    }

    api.load_model(booster.handle, options.model_path);
    const auto training_scores = api.predict(booster.handle, training_matrix.handle);
    const auto validation_scores = api.predict(booster.handle, validation_matrix.handle);
    const auto threshold = choose_threshold(validation_scores, split_data[1].labels,
                                            options.recall_target);
    MatrixGuard test_matrix{api, api.create_matrix(
        split_data[2].features, split_data[2].rows(), kFeatureCount)};
    api.set_labels(test_matrix.handle, split_data[2].labels);
    const auto test_scores = api.predict(booster.handle, test_matrix.handle);
    const std::array<std::vector<float>, 3> split_scores = {
        training_scores, validation_scores, test_scores
    };
    std::array<MetricSummary, 3> split_metrics;
    std::array<double, 3> split_average_precisions;
    for (std::size_t split = 0; split < split_data.size(); ++split) {
        split_metrics[split] =
            evaluate(split_data[split], split_scores[split], threshold);
        split_average_precisions[split] =
            average_precision(split_scores[split], split_data[split].labels);
    }
    write_edge_metrics(options.dataset_output_directory, split_metrics,
                       split_average_precisions);
    const auto metadata_path = options.model_path + ".meta";
    write_metadata(metadata_path, threshold, best_iteration, best_average_precision,
                   options.recall_target, options.seed, options.sample_factor,
                   options.rounds, options.early_stopping_rounds, options.threads,
                   scale_pos_weight);

    auto write_group_list = [&split_names](const char* split_name, std::size_t split) {
        std::cout << split_name << "_instances=";
        for (const auto& name : split_names[split]) {
            std::cout << name << ' ';
        }
        std::cout << '\n';
    };
    auto print_metrics = [](const char* title, const MetricSummary& metrics) {
        std::cout << title << ": recall=" << metrics.recall
                  << " precision=" << metrics.precision
                  << " retained_edge_fraction=" << metrics.retained_fraction
                  << " complete_tour_fraction=" << metrics.complete_tour_fraction
                  << " TP=" << metrics.true_positive << " FP=" << metrics.false_positive
                  << " FN=" << metrics.false_negative << " TN=" << metrics.true_negative
                  << '\n';
    };
    std::cout << std::setprecision(8) << "model=" << options.model_path
              << "\nmetadata=" << metadata_path
              << "\nbest_iteration=" << best_iteration
              << "\nvalidation_average_precision=" << best_average_precision
              << "\nselected_threshold=" << threshold << '\n';
    print_metrics("train", split_metrics[0]);
    print_metrics("validation", split_metrics[1]);
    print_metrics("test", split_metrics[2]);
    write_group_list("train", 0);
    write_group_list("validation", 1);
    write_group_list("test", 2);
}

void reduce_instance(const ReduceOptions& options) {
    if (options.solver_time_limit_seconds <= 0.0 ||
        !std::isfinite(options.solver_time_limit_seconds)) {
        throw std::runtime_error("Solver time limit must be finite and positive");
    }
    const auto started = std::chrono::steady_clock::now();
    const auto graph = Graph::read_tsplib(options.tsp_path);
    const auto graph_read_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const auto sampling_started = std::chrono::steady_clock::now();
    auto result = compute_features(graph, options.sample_factor, options.seed);
    const auto sampling_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sampling_started).count();
    const auto reduction_started = std::chrono::steady_clock::now();
    XGBoostApi api(get_library_path(options.xgboost_library));
    std::vector<float> values;
    values.reserve(result.edges.size() * kFeatureCount);
    for (const auto& edge : result.edges) {
        values.insert(values.end(), edge.features.begin(), edge.features.end());
    }
    MatrixGuard matrix{api, api.create_matrix(values, result.edges.size(), kFeatureCount)};
    BoosterGuard booster{api, api.create_booster(matrix.handle)};
    api.load_model(booster.handle, options.model_path);
    api.set_parameter(booster.handle, "nthread", "1");
    const auto scores = api.predict(booster.handle, matrix.handle);
    const auto metadata = read_metadata(options.model_path + ".meta");
    if (scores.size() != result.edges.size()) {
        throw std::runtime_error("Prediction count does not match number of edges");
    }
    for (std::size_t index = 0; index < result.edges.size(); ++index) {
        auto& edge = result.edges[index];
        edge.score = scores[index];
        edge.predicted_keep = edge.score >= metadata.threshold;
        edge.keep = edge.predicted_keep;
    }
    std::vector<bool> sampled_tour_edges(graph.dimension * graph.dimension, false);
    for (std::size_t i = 0; i < result.best_sampled_tour.size(); ++i) {
        const auto a = result.best_sampled_tour[i];
        const auto b = result.best_sampled_tour[(i + 1) % result.best_sampled_tour.size()];
        sampled_tour_edges[std::min(a, b) * graph.dimension + std::max(a, b)] = true;
    }
    for (auto& edge : result.edges) {
        if (sampled_tour_edges[edge.i * graph.dimension + edge.j]) {
            edge.keep = true;
        }
    }
    const auto reduction_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_started).count();
    const auto solver = solve_reduced_tsp(
        graph, result.edges, result.best_sampled_tour,
        options.solver_time_limit_seconds, options.solver_threads);
    const auto total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const auto instance_name = basename_stem(options.tsp_path);
    write_results(options.output_directory, instance_name, graph, result,
                  result.edges, metadata.threshold, graph_read_seconds,
                  sampling_seconds, reduction_seconds, total_seconds, solver);
    std::cout << "Reduced graph exported to " << options.output_directory
              << "\nRetained " << std::count_if(
                     result.edges.begin(), result.edges.end(),
                     [](const Edge& edge) { return edge.keep; })
              << " of " << result.edges.size() << " undirected edges."
              << "\nsolver_status=" << solver.status
              << "\nsolver_has_solution=" << (solver.has_solution ? "true" : "false");
    if (solver.has_solution) {
        std::cout << "\nsolver_objective=" << solver.objective
                  << "\nsolver_optimal=" << (solver.optimal ? "true" : "false");
    }
    std::cout << std::endl;
}

}  // namespace tspxgb
