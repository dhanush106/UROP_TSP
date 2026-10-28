#ifndef TSP_XGBOOST_ML_PIPELINE_H
#define TSP_XGBOOST_ML_PIPELINE_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace tspxgb {

struct TrainOptions {
    std::string data_directory;
    std::string model_path;
    std::string dataset_output_directory = "data/splits";
    std::string xgboost_library;
    std::size_t sample_factor = 100;
    std::uint64_t seed = 12345;
    std::size_t rounds = 1200;
    std::size_t early_stopping_rounds = 50;
    double recall_target = 0.995;
    std::size_t threads = 0;
};

struct ReduceOptions {
    std::string tsp_path;
    std::string model_path;
    std::string output_directory;
    std::string xgboost_library;
    std::size_t sample_factor = 100;
    std::uint64_t seed = 54321;
    double solver_time_limit_seconds = 1000.0;
    std::size_t solver_threads = 0;
};

void train_model(const TrainOptions& options);
void reduce_instance(const ReduceOptions& options);

}  // namespace tspxgb

#endif
