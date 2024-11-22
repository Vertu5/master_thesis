#ifndef CHAMP_VECTORIEL_GENERATEUR_H
#define CHAMP_VECTORIEL_GENERATEUR_H

#include <Eigen/Dense>
#include <algorithm>
#include <omp.h>
#include "ChampVectorielData.h"
#include "ChampVectorielConcatenateur.h"
#include "data_loader.h"

// Structure pour retourner les deux champs
struct DualFieldsData {
    Eigen::ArrayXXd O;  // Normalized density field
    ChampVectorielData velocity;  // Vector field (Ux, Uy)
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> mask;  // Shared mask
};

class ChampVectorielGenerateur {
public:
    static constexpr double ARENA_SIZE = 2 * 1.231;
    
    // Méthode principale pour générer les champs pour une mission complète
    static DualFieldsData generate_mission_fields(
        const std::vector<RunData>& mission_data,
        double bin_width
    ) {
        Eigen::MatrixXd all_x = ChampVectorielConcatenateur::concatenate_positions(mission_data, true);
        Eigen::MatrixXd all_y = ChampVectorielConcatenateur::concatenate_positions(mission_data, false);
        Eigen::MatrixXd all_vx = ChampVectorielConcatenateur::concatenate_velocities(mission_data, true);
        Eigen::MatrixXd all_vy = ChampVectorielConcatenateur::concatenate_velocities(mission_data, false);
        
        return generate_dual_fields(all_x, all_y, all_vx, all_vy, bin_width);
    }

    // Méthode pour générer uniquement le champ vectoriel pour un seul run
    static ChampVectorielData generate_single_run_vector_field(
        const RunData& run_data,
        double bin_width
    ) {
        // Création des matrices avec allocation parallèle
        Eigen::MatrixXd x = Eigen::MatrixXd::Zero(run_data.num_robots, run_data.num_time_steps);
        Eigen::MatrixXd y = Eigen::MatrixXd::Zero(run_data.num_robots, run_data.num_time_steps);
        Eigen::MatrixXd vx = Eigen::MatrixXd::Zero(run_data.num_robots, run_data.num_time_steps - 1);
        Eigen::MatrixXd vy = Eigen::MatrixXd::Zero(run_data.num_robots, run_data.num_time_steps - 1);

        const size_t block_size = 256;
        const size_t num_blocks = (run_data.num_time_steps + block_size - 1) / block_size;

        #pragma omp parallel for schedule(dynamic)
        for(size_t block = 0; block < num_blocks; ++block) {
            const size_t start_t = block * block_size;
            const size_t end_t = std::min(start_t + block_size, run_data.num_time_steps);

            for(size_t t = start_t; t < end_t; ++t) {
                for(size_t r = 0; r < run_data.num_robots; ++r) {
                    x(r, t) = run_data.x[t][r];
                    y(r, t) = run_data.y[t][r];
                    if(t < run_data.num_time_steps - 1) {
                        vx(r, t) = run_data.vx[t][r];
                        vy(r, t) = run_data.vy[t][r];
                    }
                }
            }
        }

        auto fields = generate_dual_fields(x, y, vx, vy, bin_width);
        return fields.velocity;
    }

private:
    static DualFieldsData generate_dual_fields(
        const Eigen::MatrixXd& x,
        const Eigen::MatrixXd& y,
        const Eigen::MatrixXd& vx,
        const Eigen::MatrixXd& vy,
        double w
    ) {
        // Transform coordinates
        Eigen::MatrixXd x_image = -y.array() + ARENA_SIZE/2;
        Eigen::MatrixXd y_image = -x.array() + ARENA_SIZE/2;
        Eigen::MatrixXd vx_image = -vy;
        Eigen::MatrixXd vy_image = -vx;

        int nx_bins = int(2 * std::ceil(0.5 * ARENA_SIZE / w));
        int ny_bins = nx_bins;

        double x_diff = nx_bins * w - ARENA_SIZE;
        double y_diff = ny_bins * w - ARENA_SIZE;
        Eigen::VectorXd x_bins = Eigen::VectorXd::LinSpaced(nx_bins + 1, -x_diff/2, ARENA_SIZE + x_diff/2);
        Eigen::VectorXd y_bins = Eigen::VectorXd::LinSpaced(ny_bins + 1, -y_diff/2, ARENA_SIZE + y_diff/2);

        // Initialize thread-local storage
        std::vector<Eigen::ArrayXXi> M_local(omp_get_max_threads(), Eigen::ArrayXXi::Zero(ny_bins, nx_bins));
        std::vector<Eigen::ArrayXXd> V_sum_x_local(omp_get_max_threads(), Eigen::ArrayXXd::Zero(ny_bins, nx_bins));
        std::vector<Eigen::ArrayXXd> V_sum_y_local(omp_get_max_threads(), Eigen::ArrayXXd::Zero(ny_bins, nx_bins));

        const size_t block_size = 8192;
        const size_t num_blocks = (vx_image.cols() + block_size - 1) / block_size;

        // Compute total number of points for normalization
        const size_t total_points = x.rows() * x.cols();
        const double bin_area = w * w;  // Area of each bin

        // Parallel accumulation with cache optimization
        #pragma omp parallel for schedule(dynamic)
        for(size_t block = 0; block < num_blocks; ++block) {
            const size_t start_t = block * block_size;
            const size_t end_t = std::min(start_t + block_size, static_cast<size_t>(vx_image.cols()));
            const int thread_id = omp_get_thread_num();

            for(size_t t = start_t; t < end_t; ++t) {
                for(int r = 0; r < vx_image.rows(); ++r) {
                    int x_idx = std::lower_bound(x_bins.data(), x_bins.data() + nx_bins + 1, 
                        x_image(r,t)) - x_bins.data() - 1;
                    int y_idx = std::lower_bound(y_bins.data(), y_bins.data() + ny_bins + 1, 
                        y_image(r,t)) - y_bins.data() - 1;

                    if(x_idx >= 0 && x_idx < nx_bins && y_idx >= 0 && y_idx < ny_bins) {
                        M_local[thread_id](y_idx, x_idx) += 1;
                        V_sum_x_local[thread_id](y_idx, x_idx) += vx_image(r,t);
                        V_sum_y_local[thread_id](y_idx, x_idx) += vy_image(r,t);
                    }
                }
            }
        }

        // Merge thread-local results
        Eigen::ArrayXXi M = Eigen::ArrayXXi::Zero(ny_bins, nx_bins);
        Eigen::ArrayXXd V_sum_x = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);
        Eigen::ArrayXXd V_sum_y = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);

        #pragma omp parallel for collapse(2)
        for(int i = 0; i < ny_bins; ++i) {
            for(int j = 0; j < nx_bins; ++j) {
                for(int t = 0; t < omp_get_max_threads(); ++t) {
                    M(i,j) += M_local[t](i,j);
                    V_sum_x(i,j) += V_sum_x_local[t](i,j);
                    V_sum_y(i,j) += V_sum_y_local[t](i,j);
                }
            }
        }

        // Create result with both density and velocity fields
        DualFieldsData result;
        
        // Normalize density field:
        // - First convert to double
        // - Then divide by (total points * bin area) to get density per unit area
        result.O = M.cast<double>() / (total_points * bin_area);
        
        result.mask = M.array() > 0;

        // Initialize velocity field
        result.velocity.Ux = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);
        result.velocity.Uy = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);
        result.velocity.mask = result.mask;  // Same mask for both fields

        // Calculate final velocity field
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < ny_bins; ++i) {
            for(int j = 0; j < nx_bins; ++j) {
                if(result.mask(i,j)) {
                    result.velocity.Ux(i,j) = V_sum_x(i,j) / M(i,j);
                    result.velocity.Uy(i,j) = V_sum_y(i,j) / M(i,j);
                }
            }
        }

        return result;
    }
};

#endif // CHAMP_VECTORIEL_GENERATEUR_H