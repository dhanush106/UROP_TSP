#include "solver.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_highs_solves_reduced_tsp() {
    tspxgb::Graph graph;
    graph.name = "square";
    graph.dimension = 4;
    graph.costs = {
        0.0, 1.0, 1.4142135623730951, 1.0,
        1.0, 0.0, 1.0, 1.4142135623730951,
        1.4142135623730951, 1.0, 0.0, 1.0,
        1.0, 1.4142135623730951, 1.0, 0.0,
    };
    std::vector<tspxgb::Edge> edges;
    for (std::size_t i = 0; i < graph.dimension; ++i) {
        for (std::size_t j = i + 1; j < graph.dimension; ++j) {
            tspxgb::Edge edge;
            edge.i = i;
            edge.j = j;
            edge.cost = graph.cost(i, j);
            edge.keep = true;
            edges.push_back(edge);
        }
    }
    const std::vector<std::size_t> warm_start = {2, 3, 0, 1};
    const auto result =
        tspxgb::solve_reduced_tsp(graph, edges, warm_start, 10.0, 1);
    require(result.has_solution, "HiGHS did not return a feasible incumbent");
    require(result.optimal, "HiGHS did not prove the square TSP optimal");
    require(std::abs(result.objective - 4.0) < 1e-8,
            "HiGHS returned an incorrect square TSP objective");
    require(std::abs(result.best_bound - 4.0) < 1e-8,
            "HiGHS returned an incorrect square TSP dual bound");
    require(std::abs(result.relative_gap) < 1e-8,
            "HiGHS returned an incorrect square TSP optimality gap");
    require(result.tour.size() == graph.dimension,
            "HiGHS returned an incomplete tour");
    require(std::abs(tspxgb::tour_cost(graph, result.tour) -
                     result.objective) < 1e-8,
            "Reported objective does not equal the returned tour cost");
}

}  // namespace

int main() {
    try {
        test_highs_solves_reduced_tsp();
        std::cout << "HiGHS solver test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Solver test failed: " << error.what() << '\n';
        return 1;
    }
}
