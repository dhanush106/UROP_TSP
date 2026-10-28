#include "solver.h"

#include <Highs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tspxgb {
namespace {

using Arc = std::pair<std::size_t, std::size_t>;

void check_highs_status(HighsStatus status, const std::string& action) {
    if (status == HighsStatus::kError) {
        throw std::runtime_error("HiGHS failed while " + action);
    }
}

}  // namespace

SolverResult solve_reduced_tsp(const Graph& graph,
                               const std::vector<Edge>& edges,
                               const std::vector<std::size_t>& warm_start,
                               double time_limit_seconds,
                               std::size_t threads) {
    const auto n = graph.dimension;
    if (n < 3 || time_limit_seconds <= 0.0 ||
        !std::isfinite(time_limit_seconds) || warm_start.size() != n) {
        throw std::runtime_error("Invalid HiGHS time limit or warm-start tour");
    }

    const auto started = std::chrono::steady_clock::now();
    std::map<Arc, HighsInt> arc_columns;

    HighsLp lp;
    lp.sense_ = ObjSense::kMinimize;
    lp.offset_ = 0.0;
    lp.model_name_ = "reduced_tsp";

    for (const auto& edge : edges) {
        if (!edge.keep) {
            continue;
        }
        if (edge.i >= n || edge.j >= n || edge.i == edge.j ||
            !std::isfinite(edge.cost)) {
            throw std::runtime_error("Invalid retained edge in reduced graph");
        }
        for (const auto& arc : {Arc(edge.i, edge.j), Arc(edge.j, edge.i)}) {
            if (arc_columns.find(arc) != arc_columns.end()) {
                continue;
            }
            const auto column = static_cast<HighsInt>(lp.col_cost_.size());
            arc_columns.emplace(arc, column);
            lp.col_cost_.push_back(edge.cost);
            lp.col_lower_.push_back(0.0);
            lp.col_upper_.push_back(1.0);
            lp.integrality_.push_back(HighsVarType::kInteger);
        }
    }

    const auto arc_count = lp.col_cost_.size();
    if (arc_count == 0) {
        throw std::runtime_error("Reduced graph has no retained edges");
    }
    const auto order_column = [arc_count](std::size_t city) {
        return static_cast<HighsInt>(arc_count + city - 1);
    };
    for (std::size_t city = 1; city < n; ++city) {
        lp.col_cost_.push_back(0.0);
        lp.col_lower_.push_back(1.0);
        lp.col_upper_.push_back(static_cast<double>(n - 1));
        lp.integrality_.push_back(HighsVarType::kInteger);
    }
    lp.num_col_ = static_cast<HighsInt>(lp.col_cost_.size());

    const auto degree_rows = 2 * n;
    const auto mtz_rows = (n - 1) * (n - 2);
    const auto row_count = degree_rows + mtz_rows;
    lp.num_row_ = static_cast<HighsInt>(row_count);
    lp.row_lower_.assign(row_count, -kHighsInf);
    lp.row_upper_.assign(row_count, kHighsInf);
    for (std::size_t city = 0; city < n; ++city) {
        lp.row_lower_[city] = 1.0;
        lp.row_upper_[city] = 1.0;
        lp.row_lower_[n + city] = 1.0;
        lp.row_upper_[n + city] = 1.0;
    }
    for (std::size_t row = degree_rows; row < row_count; ++row) {
        lp.row_upper_[row] = static_cast<double>(n - 2);
    }

    std::vector<std::vector<std::pair<HighsInt, double>>> column_entries(
        lp.col_cost_.size());
    for (const auto& item : arc_columns) {
        const auto i = item.first.first;
        const auto j = item.first.second;
        const auto column = item.second;
        column_entries[static_cast<std::size_t>(column)].push_back(
            {static_cast<HighsInt>(i), 1.0});
        column_entries[static_cast<std::size_t>(column)].push_back(
            {static_cast<HighsInt>(n + j), 1.0});
        if (i != 0 && j != 0) {
            const auto row = degree_rows + (i - 1) * (n - 2) +
                             (j < i ? j - 1 : j - 2);
            column_entries[static_cast<std::size_t>(column)].push_back(
                {static_cast<HighsInt>(row), static_cast<double>(n - 1)});
        }
    }
    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t j = 1; j < n; ++j) {
            if (i == j) {
                continue;
            }
            const auto row = degree_rows + (i - 1) * (n - 2) +
                             (j < i ? j - 1 : j - 2);
            column_entries[static_cast<std::size_t>(order_column(i))].push_back(
                {static_cast<HighsInt>(row), 1.0});
            column_entries[static_cast<std::size_t>(order_column(j))].push_back(
                {static_cast<HighsInt>(row), -1.0});
        }
    }

    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.start_.reserve(lp.col_cost_.size() + 1);
    for (auto& entries : column_entries) {
        std::sort(entries.begin(), entries.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.first < rhs.first;
                  });
        for (const auto& entry : entries) {
            lp.a_matrix_.index_.push_back(entry.first);
            lp.a_matrix_.value_.push_back(entry.second);
        }
        lp.a_matrix_.start_.push_back(
            static_cast<HighsInt>(lp.a_matrix_.index_.size()));
    }

    std::vector<bool> warm_start_seen(n, false);
    auto normalized_warm_start = warm_start;
    for (const auto city : normalized_warm_start) {
        if (city >= n || warm_start_seen[city]) {
            throw std::runtime_error("Warm-start tour is not a city permutation");
        }
        warm_start_seen[city] = true;
    }
    const auto depot = std::find(normalized_warm_start.begin(),
                                 normalized_warm_start.end(), 0);
    std::rotate(normalized_warm_start.begin(), depot,
                normalized_warm_start.end());

    Highs highs;
    check_highs_status(highs.setOptionValue("output_flag", false),
                       "disabling console output");
    check_highs_status(highs.setOptionValue("time_limit", time_limit_seconds),
                       "setting the time limit");
    if (threads > 0) {
        if (threads > static_cast<std::size_t>(
                          std::numeric_limits<HighsInt>::max())) {
            throw std::runtime_error("HiGHS thread count is too large");
        }
        check_highs_status(
            highs.setOptionValue("threads", static_cast<HighsInt>(threads)),
            "setting the thread count");
    }
    check_highs_status(highs.passModel(lp), "loading the TSP model");

    HighsSolution initial_solution;
    initial_solution.value_valid = true;
    initial_solution.col_value.assign(lp.col_cost_.size(), 0.0);
    for (std::size_t index = 0; index < normalized_warm_start.size(); ++index) {
        const auto from = normalized_warm_start[index];
        const auto to =
            normalized_warm_start[(index + 1) % normalized_warm_start.size()];
        const auto arc = arc_columns.find({from, to});
        if (arc == arc_columns.end()) {
            throw std::runtime_error(
                "Warm-start tour contains an edge removed from the reduced graph");
        }
        initial_solution.col_value[static_cast<std::size_t>(arc->second)] = 1.0;
        if (to != 0) {
            initial_solution.col_value[static_cast<std::size_t>(
                order_column(to))] = static_cast<double>(index + 1);
        }
    }
    check_highs_status(highs.setSolution(initial_solution),
                       "setting the sampled-tour MIP start");

    const auto run_status = highs.run();
    check_highs_status(run_status, "solving the reduced TSP");

    SolverResult result;
    const auto model_status = highs.getModelStatus();
    result.status = highs.modelStatusToString(model_status);
    result.optimal = model_status == HighsModelStatus::kOptimal;
    const auto& info = highs.getInfo();
    const auto& solution = highs.getSolution();
    if (solution.value_valid &&
        info.primal_solution_status == kSolutionStatusFeasible) {
        std::vector<std::size_t> next_city(n, n);
        for (const auto& item : arc_columns) {
            if (solution.col_value[static_cast<std::size_t>(item.second)] > 0.5) {
                if (next_city[item.first.first] != n) {
                    throw std::runtime_error(
                        "HiGHS incumbent has multiple outgoing tour arcs");
                }
                next_city[item.first.first] = item.first.second;
            }
        }

        std::vector<bool> visited(n, false);
        std::size_t current = 0;
        result.tour.reserve(n);
        for (std::size_t step = 0; step < n; ++step) {
            if (current >= n || visited[current] || next_city[current] == n) {
                throw std::runtime_error(
                    "HiGHS incumbent does not form a Hamiltonian cycle");
            }
            visited[current] = true;
            result.tour.push_back(current);
            current = next_city[current];
        }
        if (current != 0 ||
            std::find(visited.begin(), visited.end(), false) != visited.end()) {
            throw std::runtime_error(
                "HiGHS incumbent does not form one Hamiltonian cycle");
        }

        result.has_solution = true;
        result.objective = tour_cost(graph, result.tour);
        if (std::abs(result.objective - info.objective_function_value) >
            1e-6 * std::max(1.0, std::abs(result.objective))) {
            throw std::runtime_error(
                "HiGHS objective does not match the extracted tour cost");
        }
        if (std::isfinite(info.mip_dual_bound)) {
            result.best_bound = info.mip_dual_bound;
        }
        if (std::isfinite(info.mip_gap) && info.mip_gap >= 0.0) {
            result.relative_gap = info.mip_gap;
        } else if (result.optimal) {
            result.relative_gap = 0.0;
        }
    }
    result.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

}  // namespace tspxgb
