#ifndef TSP_XGBOOST_API_H
#define TSP_XGBOOST_API_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace tspxgb {

using DMatrixHandle = void*;
using BoosterHandle = void*;
using XGBULong = std::uint64_t;

class XGBoostApi {
public:
    explicit XGBoostApi(const std::string& library_path);
    ~XGBoostApi();
    XGBoostApi(const XGBoostApi&) = delete;
    XGBoostApi& operator=(const XGBoostApi&) = delete;

    DMatrixHandle create_matrix(const std::vector<float>& data,
                                std::size_t rows,
                                std::size_t columns) const;
    void set_labels(DMatrixHandle matrix, const std::vector<float>& labels) const;
    void free_matrix(DMatrixHandle matrix) const;

    BoosterHandle create_booster(DMatrixHandle training_matrix) const;
    void set_parameter(BoosterHandle booster,
                       const std::string& name,
                       const std::string& value) const;
    void update(BoosterHandle booster, int iteration, DMatrixHandle training_matrix) const;
    std::vector<float> predict(BoosterHandle booster, DMatrixHandle matrix) const;
    void save_model(BoosterHandle booster, const std::string& path) const;
    void load_model(BoosterHandle booster, const std::string& path) const;
    void free_booster(BoosterHandle booster) const;

private:
    void* library_ = nullptr;

    int (*create_from_mat_)(const float*, XGBULong, XGBULong, float, DMatrixHandle*) = nullptr;
    int (*set_float_info_)(DMatrixHandle, const char*, const float*, XGBULong) = nullptr;
    int (*free_matrix_)(DMatrixHandle) = nullptr;
    int (*create_booster_)(const DMatrixHandle[], XGBULong, BoosterHandle*) = nullptr;
    int (*set_param_)(BoosterHandle, const char*, const char*) = nullptr;
    int (*update_one_iter_)(BoosterHandle, int, DMatrixHandle) = nullptr;
    int (*predict_)(BoosterHandle, DMatrixHandle, int, unsigned, int,
                    XGBULong*, const float**) = nullptr;
    int (*save_model_)(BoosterHandle, const char*) = nullptr;
    int (*load_model_)(BoosterHandle, const char*) = nullptr;
    int (*free_booster_)(BoosterHandle) = nullptr;
    const char* (*get_last_error_)() = nullptr;

    void check(int status, const char* call) const;
    void* resolve(const char* symbol) const;
};

}  // namespace tspxgb

#endif
