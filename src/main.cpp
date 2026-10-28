#include "ml_pipeline.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_help() {
    std::cout
        << "TSP_XGBOOST - native C++ XGBoost edge reduction for symmetric TSP\n\n"
        << "Usage:\n"
        << "  TSP_XGBOOST train --data DIR --model FILE [options]\n"
        << "  TSP_XGBOOST reduce --tsp FILE --model FILE --output DIR [options]\n\n"
        << "Train options:\n"
        << "  --data DIR                 Folder containing matching .tsp and tour files\n"
        << "  --model FILE               Output model path (JSON recommended)\n"
        << "  --dataset-output DIR       Persistent train/validation/test data (default data/splits)\n"
        << "  --xgboost-lib FILE         Native XGBoost shared library; or set XGBOOST_LIB\n"
        << "  --sample-factor N          Random tours per city (default 100)\n"
        << "  --seed N                   Random seed (default 12345)\n"
        << "  --rounds N                 Maximum boosting rounds (default 1200)\n"
        << "  --early-stopping N         Stop after this many rounds without AP improvement\n"
        << "  --recall-target R           Validation edge-recall target (default 0.995)\n"
        << "  --threads N                XGBoost CPU threads (default 0 = library default)\n\n"
        << "Reduce options:\n"
        << "  --tsp FILE                 TSPLIB symmetric TSP instance\n"
        << "  --model FILE               Trained XGBoost model\n"
        << "  --output DIR               Directory for edge CSVs and summary\n"
        << "  --xgboost-lib FILE         Native XGBoost shared library; or set XGBOOST_LIB\n"
        << "  --sample-factor N          Random tours per city (default 100)\n"
        << "  --seed N                   Random seed (default 54321)\n"
        << "  --solver-time-limit SEC    HiGHS wall limit (default 1000)\n"
        << "  --solver-threads N         HiGHS threads (default 0 = library default)\n";
}

std::string required_value(int& index, int argc, char** argv, const std::string& option) {
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value after " + option);
    }
    return argv[++index];
}

std::size_t parse_size(const std::string& value,
                       const std::string& option,
                       bool allow_zero = false) {
    std::size_t parsed = 0;
    try {
        parsed = static_cast<std::size_t>(std::stoull(value));
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid numeric value for " + option + ": " + value);
    }
    if (parsed == 0 && !allow_zero) {
        throw std::runtime_error(option + " must be positive");
    }
    return parsed;
}

std::uint64_t parse_seed(const std::string& value, const std::string& option) {
    try {
        return static_cast<std::uint64_t>(std::stoull(value));
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid seed for " + option + ": " + value);
    }
}

int run_train(int argc, char** argv) {
    tspxgb::TrainOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--data") {
            options.data_directory = required_value(i, argc, argv, option);
        } else if (option == "--model") {
            options.model_path = required_value(i, argc, argv, option);
        } else if (option == "--dataset-output") {
            options.dataset_output_directory = required_value(i, argc, argv, option);
        } else if (option == "--xgboost-lib") {
            options.xgboost_library = required_value(i, argc, argv, option);
        } else if (option == "--sample-factor") {
            options.sample_factor = parse_size(required_value(i, argc, argv, option), option);
        } else if (option == "--seed") {
            options.seed = parse_seed(required_value(i, argc, argv, option), option);
        } else if (option == "--rounds") {
            options.rounds = parse_size(required_value(i, argc, argv, option), option);
        } else if (option == "--early-stopping") {
            options.early_stopping_rounds =
                parse_size(required_value(i, argc, argv, option), option);
        } else if (option == "--recall-target") {
            try {
                options.recall_target = std::stod(required_value(i, argc, argv, option));
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid recall target");
            }
        } else if (option == "--threads") {
            options.threads =
                parse_size(required_value(i, argc, argv, option), option, true);
        } else if (option == "--help" || option == "-h") {
            print_help();
            return 0;
        } else {
            throw std::runtime_error("Unknown train option: " + option);
        }
    }
    if (options.data_directory.empty() || options.model_path.empty()) {
        throw std::runtime_error("train requires --data and --model");
    }
    tspxgb::train_model(options);
    return 0;
}

int run_reduce(int argc, char** argv) {
    tspxgb::ReduceOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--tsp") {
            options.tsp_path = required_value(i, argc, argv, option);
        } else if (option == "--model") {
            options.model_path = required_value(i, argc, argv, option);
        } else if (option == "--output") {
            options.output_directory = required_value(i, argc, argv, option);
        } else if (option == "--xgboost-lib") {
            options.xgboost_library = required_value(i, argc, argv, option);
        } else if (option == "--sample-factor") {
            options.sample_factor = parse_size(required_value(i, argc, argv, option), option);
        } else if (option == "--seed") {
            options.seed = parse_seed(required_value(i, argc, argv, option), option);
        } else if (option == "--solver-time-limit") {
            try {
                options.solver_time_limit_seconds =
                    std::stod(required_value(i, argc, argv, option));
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid solver time limit");
            }
        } else if (option == "--solver-threads") {
            options.solver_threads =
                parse_size(required_value(i, argc, argv, option), option, true);
        } else if (option == "--help" || option == "-h") {
            print_help();
            return 0;
        } else {
            throw std::runtime_error("Unknown reduce option: " + option);
        }
    }
    if (options.tsp_path.empty() || options.model_path.empty() ||
        options.output_directory.empty()) {
        throw std::runtime_error("reduce requires --tsp, --model, and --output");
    }
    tspxgb::reduce_instance(options);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help" ||
        std::string(argv[1]) == "-h") {
        print_help();
        return argc < 2 ? 1 : 0;
    }
    try {
        const std::string command = argv[1];
        if (command == "train") {
            return run_train(argc, argv);
        }
        if (command == "reduce") {
            return run_reduce(argc, argv);
        }
        throw std::runtime_error("Unknown command: " + command);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
