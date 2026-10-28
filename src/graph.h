#ifndef TSP_XGBOOST_GRAPH_H
#define TSP_XGBOOST_GRAPH_H

#include <cstddef>
#include <string>
#include <vector>

namespace tspxgb {

class Graph {
public:
    std::string name;
    std::size_t dimension = 0;
    std::vector<double> costs;

    double cost(std::size_t i, std::size_t j) const {
        return costs[i * dimension + j];
    }

    static Graph read_tsplib(const std::string& path);
    std::vector<std::size_t> read_tour(const std::string& path) const;
};

}  // namespace tspxgb

#endif
