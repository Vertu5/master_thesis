#ifndef BIN_WIDTH_OPTIMIZER_H
#define BIN_WIDTH_OPTIMIZER_H

#include "data_loader.h"
#include "vector_field.h"
#include <vector>

struct Metrics {
    double quality_score;
    int N_c;
    double R;
};

bool find_optimal_bin_width(const std::vector<std::vector<RunData>>& all_mission_data,
                            double& optimal_w,
                            std::vector<double>& w_range,
                            std::vector<double>& quality_scores);

#endif // BIN_WIDTH_OPTIMIZER_H
