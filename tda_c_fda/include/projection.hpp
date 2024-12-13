#ifndef PROJECTION_HPP
#define PROJECTION_HPP

#include <Eigen/Dense>
#include <vector>
#include "curve_utils.hpp"
#include "srvf.hpp"

namespace projection {

/**
 * Calcule la base normale pour une SRVF donnée
 */
inline std::vector<Eigen::MatrixXd> Basis_Normal_A(const Eigen::MatrixXd& q) {
    int n = q.rows();
    int T = q.cols();
    
    // Créer la matrice identité n x n
    Eigen::MatrixXd e = Eigen::MatrixXd::Identity(n, n);
    
    // Créer le tenseur Ev
    std::vector<Eigen::MatrixXd> Ev(n, Eigen::MatrixXd::Zero(n, T));
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < T; j++) {
            Ev[i].col(j) = e.col(i);
        }
    }
    
    // Calculer qnorm
    Eigen::VectorXd qnorm(T);
    for(int t = 0; t < T; t++) {
        qnorm(t) = q.col(t).norm();
    }
    
    // Calculer delG
    std::vector<Eigen::MatrixXd> delG;
    for(int i = 0; i < n; i++) {
        Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(n, T);
        
        // Premier terme: (q[i,:] / qnorm) * q
        for(int t = 0; t < T; t++) {
            if(qnorm(t) > 0) {
                temp.col(t) += (q(i,t) / qnorm(t)) * q.col(t);
            }
        }
        
        // Second terme: qnorm * Ev[:,:,i]
        for(int t = 0; t < T; t++) {
            temp.col(t) += qnorm(t) * Ev[i].col(t);
        }
        
        delG.push_back(temp);
    }
    
    return delG;
}

/**
 * Projette une SRVF sur l'ensemble des courbes fermées
 */
inline Eigen::MatrixXd project_curve(Eigen::MatrixXd q) {
    int n = q.rows();
    int T = q.cols();
    
    double dt = (n == 2) ? 0.35 : 0.2;
    double epsilon = 1e-6;
    int max_iter = 300;
    
    Eigen::VectorXd s = Eigen::VectorXd::LinSpaced(T, 0, 1);
    
    Eigen::MatrixXd qnew = q;
    qnew /= sqrt(srvf::innerprod_q2(qnew, qnew));
    
    Eigen::VectorXd res = Eigen::VectorXd::Ones(n);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd qnorm = Eigen::VectorXd::Zero(T);
    Eigen::VectorXd G = Eigen::VectorXd::Zero(n);
    
    int iter = 1;
    while(res.norm() > epsilon && iter <= max_iter) {
        // Jacobian
        J = 3.0 * qnew * qnew.transpose() * (1.0/T);
        J += Eigen::MatrixXd::Identity(n, n);
        
        // Compute norms
        for(int i = 0; i < T; i++) {
            qnorm(i) = qnew.col(i).norm();
        }
        
        // Compute residue
        for(int i = 0; i < n; i++) {
            // Créer un vecteur temporaire pour le produit élément par élément
            Eigen::VectorXd temp = (qnew.row(i).array() * qnorm.array()).matrix();
            G(i) = curve_utils::trapz(temp, s);
        }
        res = -G;
        
        if(res.norm() < epsilon) break;
        
        // Solve system
        Eigen::VectorXd x = J.colPivHouseholderQr().solve(res);
        
        // Update with Basis_Normal_A
        std::vector<Eigen::MatrixXd> delG = Basis_Normal_A(qnew);
        Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(n, T);
        for(int i = 0; i < n; i++) {
            temp += x(i) * delG[i] * dt;
        }
        qnew += temp;
        
        qnew /= sqrt(srvf::innerprod_q2(qnew, qnew));
        iter++;
    }
    
    return qnew;
}

/**
 * Projette un vecteur w sur l'espace tangent d'une courbe q avec une base donnée
 * @param w Vecteur à projeter
 * @param q SRVF de la courbe
 * @param basis Base normale
 * @return Vecteur projeté
 */
inline Eigen::MatrixXd project_tangent(
    const Eigen::MatrixXd& w,
    const Eigen::MatrixXd& q,
    const std::vector<Eigen::MatrixXd>& basis) 
{
    if(basis.empty()) {
        return w;  // Si pas de base, retourner w directement
    }

    int n = q.rows();
    int T = q.cols();
    
    // Initialiser le vecteur projeté
    Eigen::MatrixXd v = w;
    
    // Calculer la projection
    for(const auto& b : basis) {
        // Calcul du produit scalaire avec la base
        double dot = srvf::innerprod_q2(w, b);
        
        // Soustraire la projection
        v -= dot * b;
    }
    
    return v;
}


} // namespace projection

#endif // PROJECTION_HPP