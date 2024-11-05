#include "bin_width_optimizer.h"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>

std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result(num);
    
    if (num == 1) {
        result[0] = start;
        return result;
    }
    
    double step = (end - start) / (num - 1);
    
    for (int i = 0; i < num; i++) {
        result[i] = start + (step * i);
    }
    
    return result;
}

bool find_optimal_bin_width(const std::vector<std::vector<RunData>>& all_mission_data,
                            double& optimal_w,
                            std::vector<double>& w_range,
                            std::vector<double>& quality_scores) {
    // Define w_range
    // Create equivalent of w_range = np.linspace(0.14, 0.24, 15)
    w_range = linspace(0.14, 0.24, 15);

    size_t num_w = w_range.size();
    quality_scores.resize(num_w);

    #pragma omp parallel for
    for (size_t idx = 0; idx < num_w; ++idx) {
        double w = w_range[idx];
        std::vector<double> quality_scores_w;

        for (const auto& mission_data : all_mission_data) {
            for (const auto& run_data : mission_data) {
                Eigen::ArrayXXd Ux, Uy;
                Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> mask;

                compute_vector_field(run_data, w, Ux, Uy, mask);
                double R = compute_roughness(Ux, Uy, w);
                int N_c = compute_connections(mask);

                if (R > 1e-8) {
                    double quality_score = static_cast<double>(N_c) / R;
                    quality_scores_w.push_back(quality_score);
                }
            }
        }

        if (!quality_scores_w.empty()) {
            // Median quality score
            std::nth_element(quality_scores_w.begin(), quality_scores_w.begin() + quality_scores_w.size() / 2, quality_scores_w.end());
            double median_quality = quality_scores_w[quality_scores_w.size() / 2];
            quality_scores[idx] = median_quality;
        } else {
            quality_scores[idx] = 0.0;
        }
    }

    // Find optimal w
    auto max_it = std::max_element(quality_scores.begin(), quality_scores.end());
    if (max_it != quality_scores.end()) {
        size_t max_idx = std::distance(quality_scores.begin(), max_it);
        optimal_w = w_range[max_idx];
        return true;
    } else {
        return false;
    }
}
