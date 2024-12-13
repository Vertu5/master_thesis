#ifndef SRVF_HPP
#define SRVF_HPP

#include <Eigen/Dense>
#include <tuple>
#include <iostream>
#include <cmath>
#include "curve_utils.hpp"

namespace srvf {

/**
 * Calcule le produit interne dans l'espace SRVF
 */
inline double innerprod_q2(const Eigen::MatrixXd& q1, const Eigen::MatrixXd& q2) {
    assert(q1.cols() == q2.cols() && "Les SRVF doivent avoir le même nombre de points");
    int T = q1.cols();
    double sum = (q1.array() * q2.array()).sum();
    return sum / T;
}

/**
 * Convertit une courbe beta en son SRVF q
 */
inline std::tuple<Eigen::MatrixXd, double, double> curve_to_q(
    const Eigen::MatrixXd& beta, 
    const std::string& mode="O", 
    bool scale=true) {
    
    int n = beta.rows();
    int T = beta.cols();
    
    if (T < 2) {
        std::cerr << "Erreur: La courbe doit avoir au moins 2 points" << std::endl;
        return {Eigen::MatrixXd::Zero(n, T), 0.0, 0.0};
    }
    
    // Calcul du gradient avec 1/T comme pas
    Eigen::MatrixXd v = curve_utils::gradient(beta, 1.0 / (T-1));
    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(n, T);
    
    // Calcul du SRVF
    double min_norm = 1e-10;  // Augmenté pour éviter les divisions par zéro
    for(int i = 0; i < T; i++) {
        double L = v.col(i).norm();
        if(std::isnan(L)) {
            std::cerr << "Erreur: Norme NaN à l'index " << i << std::endl;
            return {Eigen::MatrixXd::Zero(n, T), 0.0, 0.0};
        }
        if(L > min_norm) {
            q.col(i) = v.col(i) / sqrt(L);
        } else {
            q.col(i).setZero();
        }
    }
    
    // Vérification des NaN
    if(q.array().isNaN().any()) {
        std::cerr << "Erreur: NaN dans le SRVF" << std::endl;
        return {Eigen::MatrixXd::Zero(n, T), 0.0, 0.0};
    }
    
    // Calcul des longueurs
    double len = innerprod_q2(q, q);
    if(len < min_norm) {
        std::cerr << "Erreur: Longueur trop petite" << std::endl;
        return {Eigen::MatrixXd::Zero(n, T), 0.0, 0.0};
    }
    double lenq = sqrt(len);
    
    // Mise à l'échelle si demandée
    if(scale && lenq > min_norm) {
        q = q / lenq;
        len = 1.0;
        lenq = 1.0;
    }
    
    return {q, len, lenq};
}

/**
 * Calcule le centroïde d'une courbe paramétrique
 */
inline Eigen::VectorXd calculatecentroid(const Eigen::MatrixXd& beta) {
    int n = beta.rows();
    int T = beta.cols();
    
    if (T < 2) {
        std::cerr << "Erreur: La courbe doit avoir au moins 2 points" << std::endl;
        return Eigen::VectorXd::Zero(n);
    }
    
    // Créer linspace(0,1,T)
    Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(T, 0, 1);
    
    // Calcul du gradient
    Eigen::MatrixXd betadot = curve_utils::gradient(beta, 1.0 / (T-1));
    
    // Calcul des normes et de l'intégrande
    Eigen::VectorXd normbetadot = Eigen::VectorXd::Zero(T);
    Eigen::MatrixXd integrand = Eigen::MatrixXd::Zero(n, T);
    
    double min_norm = 1e-10;
    for(int i = 0; i < T; i++) {
        normbetadot(i) = betadot.col(i).norm();
        if(normbetadot(i) > min_norm) {
            integrand.col(i) = beta.col(i) * normbetadot(i);
        }
    }
    
    // Vérification des NaN
    if(normbetadot.array().isNaN().any() || integrand.array().isNaN().any()) {
        std::cerr << "Erreur: NaN dans le calcul du centroïde" << std::endl;
        return Eigen::VectorXd::Zero(n);
    }
    
    // Intégration trapézoïdale
    double scale = curve_utils::trapz(normbetadot, t);
    if(scale < min_norm) {
        std::cerr << "Erreur: Échelle trop petite pour le centroïde" << std::endl;
        return Eigen::VectorXd::Zero(n);
    }
    
    Eigen::VectorXd centroid = curve_utils::trapz(integrand, t) / scale;
    
    // Vérification finale
    if(centroid.array().isNaN().any()) {
        std::cerr << "Erreur: NaN dans le centroïde final" << std::endl;
        return Eigen::VectorXd::Zero(n);
    }
    
    return centroid;
}

/**
 * Convertit une SRVF en courbe beta
 */
inline Eigen::MatrixXd q_to_curve(const Eigen::MatrixXd& q, double scale = 1.0) {
    int n = q.rows();
    int T = q.cols();
    
    // Mise à l'échelle de q
    Eigen::MatrixXd q_scaled = q * scale;
    
    // Calcul des normes pour chaque colonne
    Eigen::VectorXd qnorm(T);
    for(int i = 0; i < T; i++) {
        qnorm(i) = q_scaled.col(i).norm();
    }
    
    // Initialisation de beta
    Eigen::MatrixXd beta = Eigen::MatrixXd::Zero(n, T);
    
    // Pour chaque dimension
    for(int i = 0; i < n; i++) {
        // Calcul de q[i, :] * qnorm
        Eigen::VectorXd integrand = q_scaled.row(i).array() * qnorm.array();
        
        // Intégration cumulative
        double sum = 0.0;
        for(int j = 0; j < T; j++) {
            if(j > 0) {
                sum += 0.5 * (integrand(j) + integrand(j-1));
            }
            beta(i, j) = sum / T;
        }
    }
    
    return beta;
}

} // namespace srvf

#endif // SRVF_HPP