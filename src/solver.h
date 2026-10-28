#ifndef TSP_XGBOOST_SOLVER_H
#define TSP_XGBOOST_SOLVER_H

#include "features.h"
#include "graph.h"

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace tspxgb {

struct SolverResult {
    bool has_solution = false;
    bool optimal = false;
    double objective = 0.0;
    double best_bound = std::numeric_limits<double>::quiet_NaN();
    double relative_gap = std::numeric_limits<double>::quiet_NaN();
    double elapsed_seconds = 0.0;
    std::string status = "not_run";
    std::vector<std::size_t> tour;
};

SolverResult solve_reduced_tsp(const Graph& graph,
                               const std::vector<Edge>& edges,
                               const std::vector<std::size_t>& warm_start,
                               double time_limit_seconds,
                               std::size_t threads);

}  // namespace tspxgb

#endif
