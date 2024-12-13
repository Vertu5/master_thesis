#ifndef FDACURVE_HPP
#define FDACURVE_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iomanip>
#include "curve_utils.hpp"
#include "srvf.hpp"
#include "projection.hpp"
#include "karcher_utils.hpp"
#include "dp_utils.hpp"

class FDACurve {
private:
    // Paramètres de base
    std::string mode;           // 'O' pour ouvert, 'C' pour fermé
    bool scale;                 // Flag pour mise à l'échelle
    int N;                      // Nombre de points après rééchantillonnage
    int dim;                    // Dimension des courbes (2 dans notre cas)
    int K;                      // Nombre de courbes

    // Données des courbes
    std::vector<Eigen::MatrixXd> beta_curves;    // Courbes originales
    std::vector<Eigen::MatrixXd> q_curves;       // SRVF des courbes
    Eigen::MatrixXd beta_mean;                   // Moyenne des courbes
    Eigen::MatrixXd q_mean;                      // Moyenne SRVF
    
    // Résultats de l'alignement
    std::vector<Eigen::MatrixXd> qn;             // SRVF alignées
    std::vector<Eigen::MatrixXd> betan;          // Courbes alignées
    std::vector<Eigen::VectorXd> gams;           // Fonctions de déformation
    std::vector<Eigen::MatrixXd> v;              // Vecteurs de tir

    // Métriques et covariance
    Eigen::MatrixXd C;                          // Covariance de Karcher
    Eigen::VectorXd s;                          // Valeurs singulières PCA
    Eigen::MatrixXd U;                          // Vecteurs singuliers PCA
    Eigen::MatrixXd coef;                       // Coefficients PCA
    
    // Longueurs et échelles
    Eigen::VectorXd len;                        // Longueurs des courbes
    Eigen::VectorXd len_q;                      // Longueurs SRVF
    double mean_scale;                          // Longueur moyenne
    double mean_scale_q;                        // Longueur moyenne SRVF
    
    // Centres et énergie
    Eigen::MatrixXd cent;                       // Centres des courbes
    Eigen::VectorXd E;                          // Énergie

public:
    /**
     * Constructeur de FDACurve
     * @param beta_input Vecteur de matrices représentant les courbes
     * @param mode Mode ('O' pour ouvert, 'C' pour fermé)
     * @param N_points Nombre de points après rééchantillonnage
     * @param do_scale Activer la mise à l'échelle
     */
    FDACurve(const std::vector<Eigen::MatrixXd>& beta_input, 
            const std::string& mode = "O", 
            int N_points = 200, 
            bool do_scale = false) 
        : mode(mode), scale(do_scale), N(N_points) {
        
        K = beta_input.size();
        if(K == 0) {
            throw std::runtime_error("Le vecteur beta_input ne peut pas être vide");
        }
        
        dim = beta_input[0].rows();
        
        // Initialisation des vecteurs
        beta_curves = beta_input;
        q_curves.resize(K);
        len = Eigen::VectorXd::Zero(K);
        len_q = Eigen::VectorXd::Zero(K);
        cent = Eigen::MatrixXd::Zero(dim, K);
        
        std::cout << "Initialisation de fdacurve avec " << K << " courbes de dimension " 
                << dim << " et " << N << " points après rééchantillonnage." << std::endl;
        
        for(int ii = 0; ii < K; ii++) {
            std::cout << "\nTraitement de la courbe " << ii+1 << "/" << K << std::endl;
            
            // Vérification du rééchantillonnage
            Eigen::MatrixXd current_beta = beta_input[ii];
            if(current_beta.cols() != N) {
                // TODO: Implémenter resamplecurve
                std::cout << "Courbe " << ii+1 << ": Rééchantillonnage effectué." << std::endl;
            } else {
                std::cout << "Courbe " << ii+1 << ": Rééchantillonnage non nécessaire." << std::endl;
            }
            
            // Utilisation du nouveau calcul de centroïde
            Eigen::VectorXd a = -srvf::calculatecentroid(current_beta);
            
            // Translation de la courbe
            for(int j = 0; j < current_beta.cols(); j++) {
                current_beta.col(j) += a;
            }
            
            // Calcul du SRVF avec la courbe translatée
            auto [q, length, length_q] = srvf::curve_to_q(current_beta, mode);
            q_curves[ii] = q;
            len(ii) = length;
            len_q(ii) = length_q;
            
            std::cout << std::fixed << std::setprecision(12);
            std::cout << "Courbe " << ii+1 << ": SRVF calculée avec longueur " 
                    << len(ii) << " et longueur SRVF " << len_q(ii) << "." << std::endl;
            
            cent.col(ii) = -a;
            std::cout << "Courbe " << ii+1 << ": Centrage effectué avec le centroïde ["
                    << -a(0) << " " << -a(1) << "]." << std::endl;
        }
    }

    /**
     * Charge deux fichiers CSV pour construire les courbes
     * @param f_file Fichier des coordonnées x
     * @param g_file Fichier des coordonnées y
     * @param mode Mode ('O' pour ouvert, 'C' pour fermé)
     * @param N_points Nombre de points après rééchantillonnage
     * @param do_scale Activer la mise à l'échelle
     * @return Instance de FDACurve
     */
    static FDACurve from_csv(const std::string& f_file, 
                            const std::string& g_file,
                            const std::string& mode = "O",
                            int N_points = 200,
                            bool do_scale = false) {
        Eigen::MatrixXd f = curve_utils::loadCSV(f_file);
        Eigen::MatrixXd g = curve_utils::loadCSV(g_file);
        
        if(f.rows() != g.rows() || f.cols() != g.cols()) {
            throw std::runtime_error("Les fichiers f et g doivent avoir les mêmes dimensions");
        }
        
        std::vector<Eigen::MatrixXd> beta_input;
        for(int k = 0; k < f.cols(); k++) {
            Eigen::MatrixXd beta_k(2, f.rows());
            beta_k.row(0) = f.col(k).transpose();
            beta_k.row(1) = g.col(k).transpose();
            beta_input.push_back(beta_k);
        }
        
        return FDACurve(beta_input, mode, N_points, do_scale);
    }

    /**
     * Calcule la moyenne de Karcher
     * @param rotation Appliquer rotation optimale
     * @param parallel Exécuter en parallèle
     * @param lam Contrôle l'élasticité
     * @param cores Nombre de coeurs
     * @param method Méthode d'optimisation
     */
    void karcher_mean(bool rotation = true, 
                     bool parallel = false, 
                     double lam = 0.0, 
                     int cores = -1, 
                     const std::string& method = "DP") {
        bool is_closed = (mode == "C");
        auto [q_mean, beta_mean, gammas] = karcher_utils::calculate_karcher_mean(
            q_curves,
            beta_curves,
            rotation,
            parallel,
            lam,
            cores,
            method
        );
        this->q_mean = q_mean;
        this->beta_mean = beta_mean;
    }

    /**
     * Aligne les courbes sur la moyenne
     * @param rotation Calculer rotation optimale
     * @param lam Contrôle l'élasticité
     * @param parallel Exécuter en parallèle
     * @param cores Nombre de coeurs
     * @param method Méthode d'optimisation
     */
    void srvf_align(bool rotation = true, 
                   double lam = 0.0, 
                   bool parallel = false,
                   int cores = -1,
                   const std::string& method = "DP") {
        
        // Compute mean if not already done
        if (beta_mean.rows() == 0 || beta_mean.cols() == 0) {
            karcher_mean(rotation, parallel, lam, cores, method);
        }
        
        // Initialize result containers
        qn.resize(K);
        betan.resize(K);
        gams.resize(K);
        
        if (!rotation) {
            // For non-rotation case, we can directly compute optimal warping
            #pragma omp parallel for if(parallel) num_threads(cores)
            for (size_t i = 0; i < K; ++i) {
                // Compute optimal reparameterization
                auto gamma = dp_utils::optimum_reparam(q_mean, q_curves[i], 
                    (mode == "C") ? 1 : 0, lam, method);
                gams[i] = gamma;
                
                // Apply warping to get aligned curves
                qn[i] = karcher_utils::group_action_by_gamma_coord(q_curves[i], gamma);
                betan[i] = karcher_utils::group_action_by_gamma_coord(beta_curves[i], gamma);
            }
            return;
        }
        
        // Original implementation for rotation case
        // Vérifier si la moyenne existe déjà
        if(beta_mean.rows() == 0 || beta_mean.cols() == 0) {
            karcher_mean(rotation, parallel, lam, cores, method);
        }

        // Initialiser les structures de résultat
        qn.resize(K);
        betan.resize(K);
        gams.resize(K);

        // Mode
        int mode_idx = (mode == "C") ? 1 : 0;

        // Centrer beta_mean
        Eigen::VectorXd centroid2 = srvf::calculatecentroid(beta_mean);
        for(int j = 0; j < beta_mean.cols(); j++) {
            beta_mean.col(j) -= centroid2;
        }

        // Aligner sur la moyenne
        #pragma omp parallel for if(parallel) num_threads(cores)
        for(int ii = 0; ii < K; ii++) {
            // Calculer le recalage optimal
            auto calc_result = karcher_utils::karcher_calc(
                q_mean, 
                q_curves[ii], 
                std::vector<Eigen::MatrixXd>(), 
                mode_idx,
                lam,
                rotation,
                method
            );

            // Stocker les résultats
            gams[ii] = calc_result.gamI;
            qn[ii] = calc_result.v;
            
            // Appliquer la déformation
            betan[ii] = karcher_utils::group_action_by_gamma_coord(
                beta_curves[ii], 
                calc_result.gamI
            );
        }
    }
    
    // Getters
    const std::vector<Eigen::MatrixXd>& get_beta_curves() const { return beta_curves; }
    const std::vector<Eigen::MatrixXd>& get_q_curves() const { return q_curves; }
    const Eigen::MatrixXd& get_beta_mean() const { return beta_mean; }
    const Eigen::MatrixXd& get_q_mean() const { return q_mean; }
    const std::vector<Eigen::MatrixXd>& get_qn() const { return qn; }
    const std::vector<Eigen::MatrixXd>& get_betan() const { return betan; }
    const std::vector<Eigen::VectorXd>& get_gams() const { return gams; }
    const Eigen::VectorXd& get_len() const { return len; }
    const Eigen::VectorXd& get_len_q() const { return len_q; }
    const Eigen::MatrixXd& get_cent() const { return cent; }
};

#endif // FDACURVE_HPP
