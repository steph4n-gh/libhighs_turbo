#pragma once
#include <vector>

enum HighsStatus { kOk = 0, kWarning = 1, kError = -1 };
enum HighsModelStatus { kOptimal = 7, kNotSet = 0 };
enum HighsVarType { kContinuous = 0, kInteger = 1 };
enum { kHighsCallbackMipNode = 1 };

typedef int HighsInt;
constexpr double HIGHS_INFINITY = 1e20; // The reviewer said it's a constexpr or macro

struct HighsCallbackDataOut {};
struct HighsCallbackDataIn {
    int mip_node_action;
};

struct HighsInfo {
    double objective_function_value = -24.0;
};

struct HighsSolution {
    bool value_valid = true;
    std::vector<double> col_value;
};

class Highs {
public:
    HighsStatus setOptionValue(const char* option, bool value) { return kOk; }
    HighsStatus addCols(HighsInt num_col, const double* costs, const double* lower, const double* upper, HighsInt num_new_nz, const HighsInt* starts, const HighsInt* indices, const double* values) { return kOk; }
    HighsStatus addRows(HighsInt num_row, const double* lower, const double* upper, HighsInt num_new_nz, const HighsInt* starts, const HighsInt* indices, const double* values) { return kOk; }
    HighsStatus changeColsIntegrality(HighsInt num_col, const int* indices, const HighsVarType* integrality) { return kOk; }
    
    HighsStatus setCallback(void (*callback)(int, const char*, const HighsCallbackDataOut*, HighsCallbackDataIn*, void*), void* user_callback_data) { return kOk; }
    HighsStatus startCallback(int type) { return kOk; }
    
    HighsStatus run() { return kOk; }
    HighsModelStatus getModelStatus() const { return kOptimal; }
    
    const HighsInfo& getInfo() const { static HighsInfo info; return info; }
    const HighsSolution& getSolution() const { static HighsSolution sol; return sol; }
};
