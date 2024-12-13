#ifndef ENERGY_UTILS_HPP
#define ENERGY_UTILS_HPP

#include <Eigen/Dense>
#include "curve_utils.hpp"
#include "interpolation.hpp"

namespace energy_utils {

/**
 * Applique une déformation gamma à une courbe
 */
inline Eigen::MatrixXd group_action_by_gamma_coord(
    const Eigen::MatrixXd& f,
    const Eigen::VectorXd& gamma,
    bool rotation = false) {
    
    if (!rotation) {
        int n = f.cols();
        int dim = f.rows();
        
        // Points d'interpolation
        Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n, 0, 1);
        
        // Interpoler chaque dimension
        Eigen::MatrixXd fn(dim, n);
        for (int i = 0; i < dim; i++) {
            fn.row(i) = interpolation::interp1_linear(t, f.row(i), gamma);
        }
        
        return fn;
    }
    
    return f; // Cas avec rotation non implémenté
}

/**
 * Calcule l'énergie rapidement pour l'alignement de courbes
 */
inline double fast_compute_energy(
    const Eigen::MatrixXd& q1,
    const Eigen::MatrixXd& q2,
    const Eigen::VectorXd& gamma,
    double lambda = 0.0) {
    
    // Appliquer la déformation
    Eigen::MatrixXd q2n = group_action_by_gamma_coord(q2, gamma);
    
    // Calculer l'énergie
    double energy = (q1 - q2n).squaredNorm();
    
    // Ajouter le terme de régularisation si lambda > 0
    if (lambda > 0.0) {
        Eigen::VectorXd grad = curve_utils::gradient(gamma, 1.0 / (gamma.size() - 1));
        energy += lambda * grad.squaredNorm();
    }
    
    return energy;
}

} // namespace energy_utils

#endif // ENERGY_UTILS_HPP
