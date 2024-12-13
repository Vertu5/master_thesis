#ifndef CHAMP_VECTORIEL_CONCATENATEUR_H
#define CHAMP_VECTORIEL_CONCATENATEUR_H

#include <Eigen/Dense>
#include <vector>
#include <omp.h>
#include "data_loader.h"

class ChampVectorielConcatenateur {
public:
    static Eigen::MatrixXd concatenate_positions(const std::vector<RunData>& mission_data, bool is_x) {
        // Parallel pre-calculation of dimensions
        size_t total_rows = 0;
        size_t max_cols = 0;
        #pragma omp parallel for reduction(+:total_rows) reduction(max:max_cols)
        for(size_t i = 0; i < mission_data.size(); ++i) {
            total_rows += mission_data[i].num_robots;
            max_cols = std::max(max_cols, mission_data[i].num_time_steps);
        }

        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(total_rows, max_cols);

        // Calculate row offsets
        std::vector<size_t> row_offsets(mission_data.size() + 1, 0);
        for(size_t i = 0; i < mission_data.size(); ++i) {
            row_offsets[i + 1] = row_offsets[i] + mission_data[i].num_robots;
        }

        // Parallel filling with optimized blocks
        const size_t block_size = 4096;
        #pragma omp parallel
        {
            // Process by run
            #pragma omp for schedule(dynamic)
            for(size_t run_idx = 0; run_idx < mission_data.size(); ++run_idx) {
                const auto& run = mission_data[run_idx];
                const auto& data = is_x ? run.x : run.y;
                const size_t start_row = row_offsets[run_idx];
                
                // Process by temporal blocks
                for(size_t t_block = 0; t_block < run.num_time_steps; t_block += block_size) {
                    const size_t end_t = std::min(t_block + block_size, run.num_time_steps);
                    
                    // Vectorized data copy
                    #pragma omp simd
                    for(size_t t = t_block; t < end_t; ++t) {
                        for(size_t r = 0; r < run.num_robots; ++r) {
                            result(start_row + r, t) = data[t][r];
                        }
                    }
                }
            }
        }
        
        return result;
    }

    static Eigen::MatrixXd concatenate_velocities(const std::vector<RunData>& mission_data, bool is_x) {
        // Parallel pre-calculation of dimensions
        size_t total_rows = 0;
        size_t max_cols = 0;
        #pragma omp parallel for reduction(+:total_rows) reduction(max:max_cols)
        for(size_t i = 0; i < mission_data.size(); ++i) {
            total_rows += mission_data[i].num_robots;
            max_cols = std::max(max_cols, mission_data[i].num_time_steps - 1);
        }

        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(total_rows, max_cols);

        // Calculate row offsets
        std::vector<size_t> row_offsets(mission_data.size() + 1, 0);
        for(size_t i = 0; i < mission_data.size(); ++i) {
            row_offsets[i + 1] = row_offsets[i] + mission_data[i].num_robots;
        }

        // Parallel filling with optimized blocks
        const size_t block_size = 4096;
        #pragma omp parallel
        {
            // Process by run
            #pragma omp for schedule(dynamic)
            for(size_t run_idx = 0; run_idx < mission_data.size(); ++run_idx) {
                const auto& run = mission_data[run_idx];
                const auto& data = is_x ? run.vx : run.vy;
                const size_t start_row = row_offsets[run_idx];
                
                // Process by temporal blocks
                for(size_t t_block = 0; t_block < run.num_time_steps - 1; t_block += block_size) {
                    const size_t end_t = std::min(t_block + block_size, run.num_time_steps - 1);
                    
                    // Vectorized data copy
                    #pragma omp simd
                    for(size_t t = t_block; t < end_t; ++t) {
                        for(size_t r = 0; r < run.num_robots; ++r) {
                            result(start_row + r, t) = data[t][r];
                        }
                    }
                }
            }
        }
        
        return result;
    }
};

#endif // CHAMP_VECTORIEL_CONCATENATEUR_H
