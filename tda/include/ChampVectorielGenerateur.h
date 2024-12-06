#ifndef CHAMP_VECTORIEL_GENERATEUR_H
#define CHAMP_VECTORIEL_GENERATEUR_H

#include <Eigen/Dense>
#include <algorithm>
#include <omp.h>
#include "ChampVectorielData.h"
#include "data_loader.h"

struct DualFieldsData {
    Eigen::ArrayXXd O;  // Champ de densité
    ChampVectorielData velocity;  // Champ de vitesse
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> mask;  // Masque de validité
};

class ChampVectorielGenerateur {
public:
    static constexpr double ARENA_SIZE = 2 * 1.231;
    static constexpr double TIME_STEP = 0.1;

    // Méthode principale pour générer les champs d'une mission (moyenne des runs)
    static DualFieldsData generate_mission_fields(const std::vector<RunData>& mission_data, double bin_width) {
        std::vector<DualFieldsData> run_fields;
        
        // Calcul pour chaque run individuellement
        for(const auto& run : mission_data) {
            run_fields.push_back(generate_single_run_fields(run, bin_width));
        }
        
        return average_fields(run_fields);
    }

    // Méthode pour générer les champs d'un seul run
    static DualFieldsData generate_single_run_fields(const RunData& run_data, double bin_width) {
        Eigen::MatrixXd x = prepare_matrix_from_run(run_data.x);
        Eigen::MatrixXd y = prepare_matrix_from_run(run_data.y);
        Eigen::MatrixXd vx = prepare_matrix_from_run(run_data.vx);
        Eigen::MatrixXd vy = prepare_matrix_from_run(run_data.vy);

        return generate_dual_fields(x, y, vx, vy, bin_width, run_data.num_robots);
    }

private:
    // Préparation de la matrice à partir des données du run
    static Eigen::MatrixXd prepare_matrix_from_run(const std::vector<std::vector<double>>& data) {
        size_t num_time_steps = data.size();
        size_t num_robots = data[0].size();
        
        Eigen::MatrixXd matrix(num_robots, num_time_steps);
        
        #pragma omp parallel for collapse(2)
        for(size_t t = 0; t < num_time_steps; ++t) {
            for(size_t r = 0; r < num_robots; ++r) {
                matrix(r, t) = data[t][r];
            }
        }
        
        return matrix;
    }

    // Moyenne des champs de tous les runs
    static DualFieldsData average_fields(const std::vector<DualFieldsData>& fields) {
        if(fields.empty()) return DualFieldsData();
        
        DualFieldsData result;
        const size_t rows = fields[0].O.rows();
        const size_t cols = fields[0].O.cols();
        
        result.O = Eigen::ArrayXXd::Zero(rows, cols);
        result.velocity.Ux = Eigen::ArrayXXd::Zero(rows, cols);
        result.velocity.Uy = Eigen::ArrayXXd::Zero(rows, cols);
        result.mask = fields[0].mask;  // Initialisation avec le premier masque
        
        // Somme des champs
        for(const auto& field : fields) {
            result.O += field.O;
            result.velocity.Ux += field.velocity.Ux;
            result.velocity.Uy += field.velocity.Uy;
            result.mask = result.mask && field.mask;  // Intersection des masques
        }
        
        // Moyenne
        double num_fields = static_cast<double>(fields.size());
        result.O /= num_fields;
        result.velocity.Ux /= num_fields;
        result.velocity.Uy /= num_fields;
        
        return result;
    }

    // Génération des champs pour un seul run
    static DualFieldsData generate_dual_fields(
        const Eigen::MatrixXd& x,
        const Eigen::MatrixXd& y,
        const Eigen::MatrixXd& vx,
        const Eigen::MatrixXd& vy,
        double w,
        size_t num_robots
    ) {
        // Transformation des coordonnées
        Eigen::MatrixXd x_image = -y.array() + ARENA_SIZE/2;
        Eigen::MatrixXd y_image = -x.array() + ARENA_SIZE/2;
        Eigen::MatrixXd vx_image = -vy;
        Eigen::MatrixXd vy_image = -vx;

        // Calcul des dimensions de la grille
        int nx_bins = int(2 * std::ceil(0.5 * ARENA_SIZE / w));
        int ny_bins = nx_bins;
        
        // Calcul des bins
        double x_diff = nx_bins * w - ARENA_SIZE;
        double y_diff = ny_bins * w - ARENA_SIZE;
        Eigen::VectorXd x_bins = Eigen::VectorXd::LinSpaced(nx_bins + 1, -x_diff/2, ARENA_SIZE + x_diff/2);
        Eigen::VectorXd y_bins = Eigen::VectorXd::LinSpaced(ny_bins + 1, -y_diff/2, ARENA_SIZE + y_diff/2);

        // Initialisation des accumulateurs locaux pour le parallélisme
        std::vector<Eigen::ArrayXXi> M_local(omp_get_max_threads(), Eigen::ArrayXXi::Zero(ny_bins, nx_bins));
        std::vector<Eigen::ArrayXXd> V_sum_x_local(omp_get_max_threads(), Eigen::ArrayXXd::Zero(ny_bins, nx_bins));
        std::vector<Eigen::ArrayXXd> V_sum_y_local(omp_get_max_threads(), Eigen::ArrayXXd::Zero(ny_bins, nx_bins));

        // Configuration du traitement par blocs
        const size_t block_size = 8192;
        const size_t num_time_steps = vx_image.cols() / 2;
        const double bin_area = w * w;
        const size_t num_blocks = (num_time_steps + block_size - 1) / block_size;

        // Calcul parallèle des champs
        #pragma omp parallel for schedule(dynamic)
        for(size_t block = 0; block < num_blocks; ++block) {
            const size_t start_t = block * block_size;
            const size_t end_t = std::min(start_t + block_size, num_time_steps);
            const int thread_id = omp_get_thread_num();

            for(size_t t = start_t; t < end_t; ++t) {
                for(size_t r = 0; r < num_robots; ++r) {
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

        // Réduction des résultats locaux
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

        // Préparation du résultat
        DualFieldsData result;
        result.O = M.cast<double>() / (num_time_steps * num_robots * bin_area);
        result.mask = M.array() > 0;

        result.velocity.Ux = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);
        result.velocity.Uy = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);
        result.velocity.mask = result.mask;

        // Calcul des vitesses moyennes
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
