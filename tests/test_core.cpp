#include "features.h"
#include "graph.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_parse_and_features() {
    const auto root = std::string("tsp_xgboost_cpp_test_temp");
#ifdef _WIN32
    CreateDirectoryA(root.c_str(), nullptr);
#else
    mkdir(root.c_str(), 0755);
#endif
    const auto tsp_path = root + "/square.tsp";
    const auto tour_path = root + "/square.sol";
    {
        std::ofstream tsp(tsp_path.c_str());
        tsp << "NAME: square\n"
            << "TYPE: TSP\n"
            << "DIMENSION: 4\n"
            << "EDGE_WEIGHT_TYPE: EUC_2D\n"
            << "NODE_COORD_SECTION\n"
            << "1 0 0\n2 0 1\n3 1 1\n4 1 0\nEOF\n";
    }
    {
        std::ofstream tour(tour_path.c_str());
        tour << "TOUR_SECTION\n1\n2\n3\n4\n-1\nEOF\n";
    }
    const auto graph = tspxgb::Graph::read_tsplib(tsp_path);
    const auto tour = graph.read_tour(tour_path);
    require(graph.dimension == 4, "Incorrect graph dimension");
    require(graph.cost(0, 2) == 1.0, "Incorrect rounded EUC_2D distance");
    require(tour.size() == 4 && tour[3] == 3, "Incorrect tour parsing");
    const auto result_a = tspxgb::compute_features(graph, 4, 17);
    const auto result_b = tspxgb::compute_features(graph, 4, 17);
    require(result_a.edges.size() == 6, "Expected six undirected edges");
    require(result_a.best_sampled_tour == result_b.best_sampled_tour,
            "Sampling is not reproducible for a fixed seed");
    require(result_a.best_sampled_cost == result_b.best_sampled_cost,
            "Sampled tour cost is not reproducible");
    for (std::size_t i = 0; i < result_a.edges.size(); ++i) {
        require(result_a.edges[i].features.size() == tspxgb::kFeatureCount,
                "Unexpected feature count");
        for (const auto value : result_a.edges[i].features) {
            require(std::isfinite(value), "Feature value is not finite");
        }
        require(result_a.edges[i].features == result_b.edges[i].features,
                "Features are not reproducible for a fixed seed");
    }
    std::remove(tsp_path.c_str());
    std::remove(tour_path.c_str());
#ifdef _WIN32
    RemoveDirectoryA(root.c_str());
#else
    rmdir(root.c_str());
#endif
}

}  // namespace

int main() {
    try {
        test_parse_and_features();
        std::cout << "Core C++ tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
