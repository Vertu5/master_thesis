#ifndef DP_UTILS_HPP
#define DP_UTILS_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "curve_utils.hpp"
#include "energy_utils.hpp"

// Prototype de la fonction externe DP
extern "C" {
#include "DynamicProgrammingQ2.h"
}

namespace dp_utils {

/**
 * @brief Résultat de l'optimisation par programmation dynamique
 */
struct DPResult {
    Eigen::VectorXd gamma; // Fonction de déformation optimale
    double cost;           // Coût du chemin optimal
};

/**
 * @brief Fast path for computing optimal reparameterization without rotation
 *
 * @param q1 SRVF à déformer
 * @param q2 SRVF de référence
 * @param lambda Paramètre d'élasticité
 * @return Vecteur gamma optimal
 */
inline Eigen::VectorXd fast_optimum_reparam(
    const Eigen::MatrixXd& q1,
    const Eigen::MatrixXd& q2,
    double lambda = 0.0) {
    
    const int n = q1.cols();
    
    // Pre-compute distances for efficiency
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n, n);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            D(i, j) = (q1.col(i) - q2.col(j)).squaredNorm();
        }
    }
    
    // Dynamic programming without rotation
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(n, n);
    E.row(0) = D.row(0);
    
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double min_val = std::numeric_limits<double>::max();
            for (int k = 0; k <= j; ++k) {
                double val = E(i-1, k) + D(i, j);
                if (lambda > 0) {
                    val += lambda * std::pow(double(j-k)/(n-1), 2);
                }
                min_val = std::min(min_val, val);
            }
            E(i, j) = min_val;
        }
    }
    
    // Backtrack to find optimal path
    Eigen::VectorXd gamma = Eigen::VectorXd::Zero(n);
    int j = n-1;
    gamma(n-1) = 1.0;
    
    for (int i = n-2; i >= 0; --i) {
        double min_val = std::numeric_limits<double>::max();
        int best_k = j;
        
        for (int k = 0; k <= j; ++k) {
            double val = E(i, k) + D(i+1, j);
            if (lambda > 0) {
                val += lambda * std::pow(double(j-k)/(n-1), 2);
            }
            if (val < min_val) {
                min_val = val;
                best_k = k;
            }
        }
        
        j = best_k;
        gamma(i) = double(j) / (n-1);
    }
    
    return gamma;
}

/**
 * @brief Interface C++ pour la fonction de programmation dynamique C
 *
 * @param q2 SRVF à déformer
 * @param q1 SRVF de référence
 * @param lam Paramètre d'élasticité
 * @return Vecteur gamma optimal
 */
inline Eigen::VectorXd optimum_reparam_curve(const Eigen::MatrixXd& q2,
                                           const Eigen::MatrixXd& q1,
                                           double lam = 0.0) {
    int n = q1.rows();  // dimension ambiante
    int T = q1.cols();  // nombre de points
    
    // Créer vecteur temps uniformément espacé
    Eigen::VectorXd time = Eigen::VectorXd::LinSpaced(T, 0, 1);
    
    // Allouer mémoire pour les résultats
    std::vector<double> G(T);
    std::vector<double> Tout(T);
    double size;

    // Convertir vers le format C
    std::vector<double> q1_data(q1.size());
    std::vector<double> q2_data(q2.size());
    
    // Simplifier la conversion des données
    if(q1.IsRowMajor) {
        std::copy(q1.transpose().data(), q1.transpose().data() + q1.size(), q1_data.begin());
    } else {
        std::copy(q1.data(), q1.data() + q1.size(), q1_data.begin());
    }
    std::copy(q2.data(), q2.data() + q2.size(), q2_data.begin());
    
    std::vector<double> time_data(time.data(), time.data() + time.size());
    
    // Appeler la fonction C
    DynamicProgrammingQ2(
        q1_data.data(), time_data.data(),
        q2_data.data(), time_data.data(),
        n, T, T,
        time_data.data(), time_data.data(),
        T, T,
        G.data(), Tout.data(),
        &size, lam,
        7  // nbhd_dim par défaut
    );
    
    // Convertir le résultat en VectorXd
    Eigen::VectorXd gamma = Eigen::Map<Eigen::VectorXd>(G.data(), T);
    gamma = (gamma.array() - gamma(0)) / (gamma(T-1) - gamma(0));
    
    return gamma;
}

/**
 * @brief Optimized version of optimum_reparam
 *
 * @param q1 SRVF à déformer
 * @param q2 SRVF de référence
 * @param mode Mode de calcul
 * @param lambda Paramètre d'élasticité
 * @param method Méthode de calcul
 * @param rotation Rotation ou non
 * @return Vecteur gamma optimal
 */
inline Eigen::VectorXd optimum_reparam(
    const Eigen::MatrixXd& q1,
    const Eigen::MatrixXd& q2,
    int mode = 0,
    double lambda = 0.0,
    const std::string& method = "DP",
    bool rotation = false) {
    
    if (!rotation && method == "DP") {
        return fast_optimum_reparam(q1, q2, lambda);
    }
    
    // Original implementation for rotation case
    return optimum_reparam_curve(q2, q1, lambda);
}

/**
 * @brief Fast path for computing energy without rotation
 *
 * @param q1 SRVF à déformer
 * @param q2 SRVF de référence
 * @param gamma Fonction de déformation
 * @param lambda Paramètre d'élasticité
 * @return Énergie du chemin optimal
 */
inline double compute_energy(
    const Eigen::MatrixXd& q1,
    const Eigen::MatrixXd& q2,
    const Eigen::VectorXd& gamma,
    double lambda = 0.0) {
    
    return energy_utils::fast_compute_energy(q1, q2, gamma, lambda);
}

} // namespace dp_utils

#endif // DP_UTILS_HPP