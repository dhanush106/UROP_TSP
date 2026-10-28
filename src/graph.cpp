#include "graph.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace tspxgb {
namespace {

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<double> parse_explicit(const std::vector<double>& values,
                                   std::size_t n,
                                   const std::string& format) {
    std::vector<double> matrix(n * n, 0.0);
    if (format == "FULL_MATRIX") {
        if (values.size() != n * n) {
            throw std::runtime_error("FULL_MATRIX value count does not match DIMENSION");
        }
        matrix = values;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                if (std::abs(matrix[i * n + j] - matrix[j * n + i]) > 1e-9) {
                    throw std::runtime_error("Only symmetric FULL_MATRIX instances are supported");
                }
            }
        }
        return matrix;
    }

    const bool row_order = format.find("_ROW") != std::string::npos;
    const bool lower = format.find("LOWER") != std::string::npos;
    const bool diagonal = format.find("DIAG") != std::string::npos;
    if (!row_order && format.find("_COL") == std::string::npos) {
        throw std::runtime_error("Unsupported EDGE_WEIGHT_FORMAT: " + format);
    }

    std::size_t index = 0;
    auto add = [&](std::size_t i, std::size_t j) {
        if (index >= values.size()) {
            throw std::runtime_error("Not enough EDGE_WEIGHT_SECTION values");
        }
        matrix[i * n + j] = values[index];
        matrix[j * n + i] = values[index];
        ++index;
    };

    if (row_order) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const bool selected = lower ? (diagonal ? j <= i : j < i)
                                            : (diagonal ? j >= i : j > i);
                if (selected) {
                    add(i, j);
                }
            }
        }
    } else {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                const bool selected = lower ? (diagonal ? i >= j : i > j)
                                            : (diagonal ? i <= j : i < j);
                if (selected) {
                    add(i, j);
                }
            }
        }
    }
    if (index != values.size()) {
        throw std::runtime_error("Too many EDGE_WEIGHT_SECTION values");
    }
    return matrix;
}

double round_tsplib(double value) {
    return std::floor(value + 0.5);
}

std::vector<std::size_t> validate_tour(std::vector<std::size_t> tour,
                                       std::size_t dimension) {
    if (tour.size() == dimension + 1 && tour.front() == tour.back()) {
        tour.pop_back();
    }
    if (tour.size() != dimension) {
        throw std::runtime_error("Optimal tour must contain exactly DIMENSION city IDs");
    }
    std::unordered_set<std::size_t> unique(tour.begin(), tour.end());
    if (unique.size() != dimension ||
        *std::max_element(tour.begin(), tour.end()) >= dimension) {
        throw std::runtime_error("Optimal tour must be a permutation of city IDs 1..n");
    }
    return tour;
}

}  // namespace

Graph Graph::read_tsplib(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open TSP file: " + path);
    }

    std::string name;
    std::string weight_type;
    std::string weight_format;
    std::size_t dimension = 0;
    std::string section;
    std::vector<std::pair<double, double>> coordinates;
    std::vector<double> edge_weights;
    std::string line;

    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        if (line == "EOF") {
            break;
        }
        if (line == "NODE_COORD_SECTION") {
            section = "COORD";
            continue;
        }
        if (line == "EDGE_WEIGHT_SECTION") {
            section = "WEIGHT";
            continue;
        }
        if (section == "COORD") {
            std::istringstream row(line);
            std::size_t id = 0;
            double x = 0.0;
            double y = 0.0;
            if (!(row >> id >> x >> y)) {
                throw std::runtime_error("Malformed coordinate row in " + path);
            }
            coordinates.emplace_back(x, y);
            continue;
        }
        if (section == "WEIGHT") {
            std::istringstream row(line);
            double value = 0.0;
            while (row >> value) {
                edge_weights.push_back(value);
            }
            continue;
        }
        const auto delimiter = line.find(':');
        if (delimiter == std::string::npos) {
            continue;
        }
        const auto key = trim(line.substr(0, delimiter));
        const auto value = trim(line.substr(delimiter + 1));
        if (key == "NAME") {
            name = value;
        } else if (key == "DIMENSION") {
            dimension = static_cast<std::size_t>(std::stoul(value));
        } else if (key == "EDGE_WEIGHT_TYPE") {
            weight_type = value;
        } else if (key == "EDGE_WEIGHT_FORMAT") {
            weight_format = value;
        }
    }
    if (dimension < 3) {
        throw std::runtime_error("Missing or invalid DIMENSION in " + path);
    }

    Graph graph;
    graph.name = name.empty() ? path : name;
    graph.dimension = dimension;
    graph.costs.assign(dimension * dimension, 0.0);

    if (!coordinates.empty()) {
        if (coordinates.size() != dimension) {
            throw std::runtime_error("Coordinate count does not match DIMENSION in " + path);
        }
        if (weight_type != "EUC_2D" && weight_type != "CEIL_2D" &&
            weight_type != "ATT" && weight_type != "MAX_2D" &&
            weight_type != "GEO") {
            throw std::runtime_error("Unsupported coordinate EDGE_WEIGHT_TYPE: " + weight_type);
        }
        const double pi = std::acos(-1.0);
        for (std::size_t i = 0; i < dimension; ++i) {
            for (std::size_t j = i + 1; j < dimension; ++j) {
                const auto dx = coordinates[i].first - coordinates[j].first;
                const auto dy = coordinates[i].second - coordinates[j].second;
                double distance = 0.0;
                if (weight_type == "EUC_2D") {
                    distance = round_tsplib(std::sqrt(dx * dx + dy * dy));
                } else if (weight_type == "CEIL_2D") {
                    distance = std::ceil(std::sqrt(dx * dx + dy * dy));
                } else if (weight_type == "ATT") {
                    distance = std::ceil(std::sqrt((dx * dx + dy * dy) / 10.0));
                } else if (weight_type == "MAX_2D") {
                    distance = round_tsplib(std::max(std::abs(dx), std::abs(dy)));
                } else {
                    auto geo_angle = [pi](double coordinate) {
                        const auto degrees = std::floor(coordinate);
                        const auto minutes = coordinate - degrees;
                        return pi * (degrees + 5.0 * minutes / 3.0) / 180.0;
                    };
                    const auto lat_i = geo_angle(coordinates[i].first);
                    const auto lon_i = geo_angle(coordinates[i].second);
                    const auto lat_j = geo_angle(coordinates[j].first);
                    const auto lon_j = geo_angle(coordinates[j].second);
                    const auto q1 = std::cos(lon_i - lon_j);
                    const auto q2 = std::cos(lat_i - lat_j);
                    const auto q3 = std::cos(lat_i + lat_j);
                    const auto raw_argument =
                        0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3);
                    const auto argument = std::max(-1.0, std::min(1.0, raw_argument));
                    distance = std::floor(6378.388 * std::acos(argument) + 1.0);
                }
                graph.costs[i * dimension + j] = distance;
                graph.costs[j * dimension + i] = distance;
            }
        }
    } else if (weight_type == "EXPLICIT") {
        graph.costs = parse_explicit(edge_weights, dimension, weight_format);
    } else {
        throw std::runtime_error("No coordinates or supported explicit matrix in " + path);
    }
    return graph;
}

std::vector<std::size_t> Graph::read_tour(const std::string& path) const {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open optimal-tour file: " + path);
    }
    std::vector<std::size_t> tour;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line == "TOUR_SECTION") {
            continue;
        }
        if (line == "EOF") {
            break;
        }
        if (line.empty()) {
            continue;
        }
        std::istringstream row(line);
        long long id = 0;
        while (row >> id) {
            if (id == -1) {
                return validate_tour(std::move(tour), dimension);
            }
            if (id > 0) {
                tour.push_back(static_cast<std::size_t>(id - 1));
            }
        }
    }
    return validate_tour(std::move(tour), dimension);
}

}  // namespace tspxgb
