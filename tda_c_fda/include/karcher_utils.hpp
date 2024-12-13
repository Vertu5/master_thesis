#ifndef KARCHER_UTILS_HPP
#define KARCHER_UTILS_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>
#include "curve_utils.hpp"
#include "srvf.hpp"
#include "dp_utils.hpp"
#include "interpolation.hpp"
#include "energy_utils.hpp"

using namespace curve_utils;  
using namespace interpolation;  

namespace karcher_utils {

// Structures de données pour les résultats
struct KarcherCalcResult {
    Eigen::MatrixXd v;     
    Eigen::VectorXd gamI;  
    double d;              
};

struct KarcherMeanResult {
    Eigen::MatrixXd q_mean;      
    Eigen::MatrixXd beta_mean;   
    Eigen::MatrixXd v;           
    Eigen::VectorXd gamI;        
    double d;                    
};

// Inverse une fonction de déformation
inline Eigen::VectorXd invertGamma(const Eigen::VectorXd& gamma) {
    int T = gamma.size();
    
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(T, 0, 1);
    Eigen::VectorXd gammaI = Eigen::VectorXd::Zero(T);
    
    for(int i = 0; i < T; i++) {
        double target = x[i];
        int idx = 0;
        while(idx < T-1 && gamma[idx+1] < target) idx++;
        
        if(idx == T-1) {
            gammaI[i] = 1.0;
        } else {
            double alpha = (target - gamma[idx]) / (gamma[idx+1] - gamma[idx]);
            gammaI[i] = x[idx] + alpha * (x[idx+1] - x[idx]);
        }
    }
    
    gammaI = (gammaI.array() - gammaI[0]) / (gammaI[T-1] - gammaI[0]);
    return gammaI;
}

// Décale une courbe
inline Eigen::MatrixXd shift_f(const Eigen::MatrixXd& f, int tau) {
    int n = f.rows();
    int T = f.cols();
    
    Eigen::MatrixXd fn = Eigen::MatrixXd::Zero(n, T);
    
    for(int i = 0; i < T-1; i++) {
        int new_idx = (i + tau) % (T-1);
        fn.col(new_idx) = f.col(i);
    }
    
    fn.col(T-1) = fn.col(0);
    return fn;
}

// Applique une déformation gamma à une courbe
inline Eigen::MatrixXd group_action_by_gamma_coord(
    const Eigen::MatrixXd& f,
    const Eigen::VectorXd& gamma,
    bool rotation = false) {
    
    return energy_utils::group_action_by_gamma_coord(f, gamma, rotation);
}

// Calcul vecteur de tir et déformation optimale
inline KarcherCalcResult karcher_calc(
    const Eigen::MatrixXd& mu,
    const Eigen::MatrixXd& q,
    const std::vector<Eigen::MatrixXd>& basis,
    int closed,
    double lam,
    bool rotation,
    const std::string& method) 
{
    KarcherCalcResult result;
    
    Eigen::VectorXd gamI = dp_utils::optimum_reparam_curve(q, mu, lam);
    result.gamI = gamI;

    Eigen::MatrixXd qn_t = group_action_by_gamma_coord(q, gamI, rotation);
    qn_t = qn_t / sqrt(srvf::innerprod_q2(qn_t, qn_t));
    
    double q1dotq2 = srvf::innerprod_q2(mu, qn_t);
    if(q1dotq2 > 1) q1dotq2 = 1;
    
    result.d = std::acos(q1dotq2);
    
    Eigen::MatrixXd u = qn_t - q1dotq2 * mu;
    double normu = sqrt(srvf::innerprod_q2(u, u));
    
    Eigen::MatrixXd w;
    if(normu > 1e-4) {
        w = u * std::acos(q1dotq2) / normu;
    } else {
        w = Eigen::MatrixXd::Zero(qn_t.rows(), qn_t.cols());
    }
    
    if(closed == 0) {
        result.v = w;
    } else {
        result.v = projection::project_tangent(w, q, basis);
    }
    
    return result;
}

/**
 * Calcul rapide de la moyenne de Karcher pour le cas non-rotatif
 */
inline std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<Eigen::VectorXd>> 
fast_calculate_karcher_mean(
    const std::vector<Eigen::MatrixXd>& q_curves,
    const std::vector<Eigen::MatrixXd>& beta_curves,
    bool parallel = false,
    double lambda = 0.0,
    int cores = -1,
    const std::string& method = "DP") {
    
    const size_t K = q_curves.size();
    if (K == 0) return {};
    
    const int n = q_curves[0].cols();
    
    // Initialize with arithmetic mean
    Eigen::MatrixXd q_mean = Eigen::MatrixXd::Zero(q_curves[0].rows(), n);
    Eigen::MatrixXd beta_mean = Eigen::MatrixXd::Zero(beta_curves[0].rows(), n);
    
    for (const auto& q : q_curves) q_mean += q;
    for (const auto& beta : beta_curves) beta_mean += beta;
    
    q_mean /= K;
    beta_mean /= K;
    
    // Initialize containers for results
    std::vector<Eigen::VectorXd> gammas(K);
    std::vector<Eigen::MatrixXd> qn(K), betan(K);
    
    // Iterative refinement
    double energy_old = std::numeric_limits<double>::max();
    const int max_iters = 20;
    
    for (int iter = 0; iter < max_iters; ++iter) {
        double energy_new = 0.0;
        
        #pragma omp parallel for if(parallel) num_threads(cores) reduction(+:energy_new)
        for (size_t i = 0; i < K; ++i) {
            // Compute optimal warping
            gammas[i] = dp_utils::optimum_reparam(q_mean, q_curves[i], 0, lambda, method);
            
            // Apply warping
            qn[i] = energy_utils::group_action_by_gamma_coord(q_curves[i], gammas[i]);
            betan[i] = energy_utils::group_action_by_gamma_coord(beta_curves[i], gammas[i]);
            
            // Accumulate energy
            energy_new += energy_utils::fast_compute_energy(q_mean, qn[i], gammas[i], lambda);
        }
        
        // Update mean
        q_mean.setZero();
        beta_mean.setZero();
        
        for (size_t i = 0; i < K; ++i) {
            q_mean += qn[i];
            beta_mean += betan[i];
        }
        q_mean /= K;
        beta_mean /= K;
        
        // Check convergence
        if (std::abs(energy_new - energy_old) < 1e-6) break;
        energy_old = energy_new;
    }
    
    return {q_mean, beta_mean, gammas};
}

// Calcule la moyenne de Karcher
inline std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<Eigen::VectorXd>> 
calculate_karcher_mean(
    const std::vector<Eigen::MatrixXd>& q_curves,
    const std::vector<Eigen::MatrixXd>& beta_curves,
    bool rotation = false,
    bool parallel = false,
    double lambda = 0.0,
    int cores = -1,
    const std::string& method = "DP") {
    
    if (!rotation) {
        return fast_calculate_karcher_mean(
            q_curves, beta_curves, parallel, lambda, cores, method);
    }
    
    int n = beta_curves[0].rows();
    int T = beta_curves[0].cols();
    int N = beta_curves.size();
    
    std::cout << "Shape of beta: n=" << n << ", T=" << T << ", N=" << N << std::endl;
    int mode_idx = (method == "C") ? 1 : 0;
    std::cout << "Mode: " << mode_idx << std::endl;
    
    KarcherMeanResult result;
    result.q_mean = q_curves[0];
    result.beta_mean = beta_curves[0];
    
    std::cout << "Computing Karcher Mean of " << N 
              << " curves in SRVF space with lam=" << lambda << std::endl;

    const int maxit = 20;
    const double tolv = 1e-4;
    const double told = 5e-3;
    const double delta = 0.5;
    int itr = 0;

    Eigen::VectorXd sumd = Eigen::VectorXd::Zero(maxit + 1);
    Eigen::VectorXd normvbar = Eigen::VectorXd::Zero(maxit + 1);
    std::vector<Eigen::MatrixXd> basis;
    sumd(0) = std::numeric_limits<double>::infinity();

    std::cout << "\nInitial q_mean norm: " 
              << sqrt(srvf::innerprod_q2(result.q_mean, result.q_mean)) << std::endl;

    while(itr < maxit) {
        std::cout << "\n========= Iteration " << itr << " =========\n";
        
        result.q_mean = result.q_mean / sqrt(srvf::innerprod_q2(result.q_mean, result.q_mean));
        
        std::cout << "q_mean norm après normalisation: " 
                  << sqrt(srvf::innerprod_q2(result.q_mean, result.q_mean)) << "\n";

        if(mode_idx == 1) {
            basis = projection::Basis_Normal_A(result.q_mean);
            std::cout << "Base calculée pour mode fermé\n";
        }

        Eigen::MatrixXd sumv = Eigen::MatrixXd::Zero(n, T);
        sumd(itr + 1) = 0;

        std::cout << "\nTraitement des courbes pour l'itération " << itr << ":\n";

        for(int i = 0; i < N; i++) {
            auto calc_result = karcher_calc(result.q_mean, q_curves[i], basis, mode_idx,
                                        lambda, rotation, method);
            
            sumv += calc_result.v;
            sumd(itr + 1) += calc_result.d * calc_result.d;

            std::cout << "Courbe " << i << ": "
                      << "v norm = " << sqrt(srvf::innerprod_q2(calc_result.v, calc_result.v))
                      << ", d = " << calc_result.d
                      << ", sumd cumul = " << sumd(itr + 1) << "\n";

            if(itr == maxit - 1) {
                result.v = calc_result.v;
                result.gamI = calc_result.gamI;
                result.d = calc_result.d;
            }
        }

        Eigen::MatrixXd vbar = sumv / static_cast<double>(N);
        normvbar(itr) = sqrt(srvf::innerprod_q2(vbar, vbar));
        double normv = normvbar(itr);

        std::cout << "\nRésumé de l'itération " << itr << ":\n"
                  << "normvbar = " << normvbar(itr) 
                  << ", sumd[" << itr << "] = " << sumd(itr) 
                  << ", sumd[" << itr+1 << "] = " << sumd(itr + 1) 
                  << ", diff = " << sumd(itr) - sumd(itr + 1) << "\n";

        if(sumd(itr) - sumd(itr + 1) < 0) {
            std::cout << "BREAK: coût croissant détecté\n";
            break;
        }
        
        if(normv > tolv && std::abs(sumd(itr + 1) - sumd(itr)) > told) {
            double angle = delta * normvbar(itr);
            result.q_mean = std::cos(angle) * result.q_mean + 
                           std::sin(angle) * vbar / normvbar(itr);

            std::cout << "Mise à jour q_mean: angle = " << angle 
                      << ", nouvelle norme = " 
                      << sqrt(srvf::innerprod_q2(result.q_mean, result.q_mean)) << "\n";

            if(mode_idx == 1) {
                result.q_mean = projection::project_curve(result.q_mean);
                std::cout << "Projection effectuée, nouvelle norme = " 
                          << sqrt(srvf::innerprod_q2(result.q_mean, result.q_mean)) << "\n";
            }

            Eigen::MatrixXd x = srvf::q_to_curve(result.q_mean);
            Eigen::VectorXd a = -srvf::calculatecentroid(x);
            result.beta_mean = x + a.replicate(1, T);
            
            std::cout << "beta_mean mis à jour, norme = " 
                      << result.beta_mean.norm() << "\n";
        } 
        else {
            std::cout << "BREAK: convergence atteinte (normv = " << normv 
                      << ", diff sumd = " << std::abs(sumd(itr + 1) - sumd(itr)) << ")\n";
            break;
        }

        itr++;
    }

    if(lambda > 0) {
        double mean_scale = std::pow(lambda, 1.0/q_curves.size());
        result.beta_mean *= mean_scale;
        std::cout << "\nMise à l'échelle appliquée avec facteur " << mean_scale << "\n";
    }

    std::cout << "\nRésultats finaux:\n"
              << "Nombre d'itérations: "<< itr << "\n"
              << "Norme q_mean finale: " 
              << sqrt(srvf::innerprod_q2(result.q_mean, result.q_mean)) << "\n"
              << "Norme beta_mean finale: " << result.beta_mean.norm() << "\n";

    return {result.q_mean, result.beta_mean, {result.gamI}};
}

} // namespace karcher_utils

#endif // KARCHER_UTILS_HPP
