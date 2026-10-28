#include "xgboost_api.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tspxgb {

XGBoostApi::XGBoostApi(const std::string& library_path) {
#ifdef _WIN32
    library_ = static_cast<void*>(
        LoadLibraryExA(library_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH));
#else
    library_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (library_ == nullptr) {
        throw std::runtime_error("Cannot load XGBoost shared library: " + library_path);
    }
    try {
        create_from_mat_ = reinterpret_cast<decltype(create_from_mat_)>(
            resolve("XGDMatrixCreateFromMat"));
        set_float_info_ = reinterpret_cast<decltype(set_float_info_)>(
            resolve("XGDMatrixSetFloatInfo"));
        free_matrix_ = reinterpret_cast<decltype(free_matrix_)>(resolve("XGDMatrixFree"));
        create_booster_ = reinterpret_cast<decltype(create_booster_)>(
            resolve("XGBoosterCreate"));
        set_param_ = reinterpret_cast<decltype(set_param_)>(resolve("XGBoosterSetParam"));
        update_one_iter_ = reinterpret_cast<decltype(update_one_iter_)>(
            resolve("XGBoosterUpdateOneIter"));
        predict_ = reinterpret_cast<decltype(predict_)>(resolve("XGBoosterPredict"));
        save_model_ = reinterpret_cast<decltype(save_model_)>(
            resolve("XGBoosterSaveModel"));
        load_model_ = reinterpret_cast<decltype(load_model_)>(
            resolve("XGBoosterLoadModel"));
        free_booster_ = reinterpret_cast<decltype(free_booster_)>(
            resolve("XGBoosterFree"));
        get_last_error_ = reinterpret_cast<decltype(get_last_error_)>(
            resolve("XGBGetLastError"));
    } catch (...) {
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(library_));
#else
        dlclose(library_);
#endif
        library_ = nullptr;
        throw;
    }
}

XGBoostApi::~XGBoostApi() {
    // Keep XGBoost loaded until process exit; some builds crash during library teardown.
}

void* XGBoostApi::resolve(const char* symbol) const {
#ifdef _WIN32
    auto address = reinterpret_cast<void*>(
        GetProcAddress(static_cast<HMODULE>(library_), symbol));
#else
    auto address = dlsym(library_, symbol);
#endif
    if (address == nullptr) {
        throw std::runtime_error(std::string("XGBoost shared library lacks symbol: ") + symbol);
    }
    return address;
}

void XGBoostApi::check(int status, const char* call) const {
    if (status != 0) {
        const auto* detail = get_last_error_ == nullptr ? nullptr : get_last_error_();
        throw std::runtime_error(std::string(call) + " failed: " +
                                 (detail == nullptr ? "unknown XGBoost error" : detail));
    }
}

DMatrixHandle XGBoostApi::create_matrix(const std::vector<float>& data,
                                        std::size_t rows,
                                        std::size_t columns) const {
    if (rows == 0 || columns == 0 || data.size() != rows * columns) {
        throw std::runtime_error("Invalid dense feature matrix dimensions");
    }
    DMatrixHandle matrix = nullptr;
    check(create_from_mat_(data.data(), static_cast<XGBULong>(rows),
                           static_cast<XGBULong>(columns),
                           std::numeric_limits<float>::quiet_NaN(), &matrix),
          "XGDMatrixCreateFromMat");
    return matrix;
}

void XGBoostApi::set_labels(DMatrixHandle matrix,
                            const std::vector<float>& labels) const {
    check(set_float_info_(matrix, "label", labels.data(),
                          static_cast<XGBULong>(labels.size())),
          "XGDMatrixSetFloatInfo");
}

void XGBoostApi::free_matrix(DMatrixHandle matrix) const {
    if (matrix != nullptr) {
        free_matrix_(matrix);
    }
}

BoosterHandle XGBoostApi::create_booster(DMatrixHandle training_matrix) const {
    BoosterHandle booster = nullptr;
    if (training_matrix == nullptr) {
        check(create_booster_(nullptr, 0, &booster), "XGBoosterCreate");
    } else {
        const DMatrixHandle matrices[] = {training_matrix};
        check(create_booster_(matrices, 1, &booster), "XGBoosterCreate");
    }
    return booster;
}

void XGBoostApi::set_parameter(BoosterHandle booster,
                               const std::string& name,
                               const std::string& value) const {
    check(set_param_(booster, name.c_str(), value.c_str()), "XGBoosterSetParam");
}

void XGBoostApi::update(BoosterHandle booster,
                        int iteration,
                        DMatrixHandle training_matrix) const {
    check(update_one_iter_(booster, iteration, training_matrix),
          "XGBoosterUpdateOneIter");
}

std::vector<float> XGBoostApi::predict(BoosterHandle booster,
                                       DMatrixHandle matrix) const {
    XGBULong length = 0;
    const float* values = nullptr;
    check(predict_(booster, matrix, 0, 0, 0, &length, &values), "XGBoosterPredict");
    if (values == nullptr && length != 0) {
        throw std::runtime_error("XGBoost returned a null prediction buffer");
    }
    if (length == 0) {
        return {};
    }
    return std::vector<float>(values, values + length);
}

void XGBoostApi::save_model(BoosterHandle booster, const std::string& path) const {
    check(save_model_(booster, path.c_str()), "XGBoosterSaveModel");
}

void XGBoostApi::load_model(BoosterHandle booster, const std::string& path) const {
    check(load_model_(booster, path.c_str()), "XGBoosterLoadModel");
}

void XGBoostApi::free_booster(BoosterHandle booster) const {
    if (booster != nullptr) {
        free_booster_(booster);
    }
}

}  // namespace tspxgb
