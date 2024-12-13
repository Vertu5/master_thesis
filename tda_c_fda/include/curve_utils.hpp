#ifndef CURVE_UTILS_HPP
#define CURVE_UTILS_HPP

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

namespace curve_utils {

/**
 * Calcule le gradient d'une matrice (similaire à numpy.gradient)
 * @param f Matrice d'entrée
 * @param dx Pas d'échantillonnage
 * @return Matrice du gradient
 */
inline Eigen::MatrixXd gradient(const Eigen::MatrixXd& f, double dx) {
    if (dx <= 0) {
        std::cerr << "Erreur: dx doit être positif" << std::endl;
        return Eigen::MatrixXd::Zero(f.rows(), f.cols());
    }
    
    int rows = f.rows();
    int cols = f.cols();
    
    if (cols < 2) {
        std::cerr << "Erreur: La matrice doit avoir au moins 2 colonnes" << std::endl;
        return Eigen::MatrixXd::Zero(rows, cols);
    }
    
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(rows, cols);
    
    // Premier point : différence avant
    grad.col(0) = (f.col(1) - f.col(0)) / dx;
    
    // Points intermédiaires : différence centrale
    for(int i = 1; i < cols-1; i++) {
        grad.col(i) = (f.col(i+1) - f.col(i-1)) / (2.0 * dx);
    }
    
    // Dernier point : différence arrière
    grad.col(cols-1) = (f.col(cols-1) - f.col(cols-2)) / dx;
    
    // Vérification des NaN
    if(grad.array().isNaN().any()) {
        std::cerr << "Erreur: NaN dans le gradient" << std::endl;
        return Eigen::MatrixXd::Zero(rows, cols);
    }
    
    return grad;
}

/**
 * Intégration trapézoïdale pour vecteur
 * @param y Vecteur des valeurs
 * @param x Vecteur des points d'échantillonnage
 * @return Valeur de l'intégrale
 */
inline double trapz(const Eigen::VectorXd& y, const Eigen::VectorXd& x) {
    if (x.size() != y.size()) {
        std::cerr << "Erreur: x et y doivent avoir la même taille" << std::endl;
        return 0.0;
    }
    
    if (x.size() < 2) {
        std::cerr << "Erreur: Au moins 2 points sont nécessaires" << std::endl;
        return 0.0;
    }
    
    double result = 0.0;
    for(int i = 0; i < x.size()-1; i++) {
        double dx = x(i+1) - x(i);
        if (dx <= 0) {
            std::cerr << "Erreur: Les points x doivent être strictement croissants" << std::endl;
            return 0.0;
        }
        result += 0.5 * dx * (y(i) + y(i+1));
    }
    
    if (std::isnan(result)) {
        std::cerr << "Erreur: NaN dans l'intégration" << std::endl;
        return 0.0;
    }
    
    return result;
}

/**
 * Intégration trapézoïdale pour matrice
 * @param y Matrice des valeurs
 * @param x Vecteur des points d'échantillonnage
 * @return Vecteur des intégrales pour chaque ligne
 */
inline Eigen::VectorXd trapz(const Eigen::MatrixXd& y, const Eigen::VectorXd& x) {
    if (x.size() != y.cols()) {
        std::cerr << "Erreur: Le nombre de colonnes de y doit correspondre à la taille de x" << std::endl;
        return Eigen::VectorXd::Zero(y.rows());
    }
    
    if (x.size() < 2) {
        std::cerr << "Erreur: Au moins 2 points sont nécessaires" << std::endl;
        return Eigen::VectorXd::Zero(y.rows());
    }
    
    int rows = y.rows();
    Eigen::VectorXd result = Eigen::VectorXd::Zero(rows);
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < x.size()-1; j++) {
            double dx = x(j+1) - x(j);
            if (dx <= 0) {
                std::cerr << "Erreur: Les points x doivent être strictement croissants" << std::endl;
                return Eigen::VectorXd::Zero(rows);
            }
            result(i) += 0.5 * dx * (y(i,j) + y(i,j+1));
        }
    }
    
    if (result.array().isNaN().any()) {
        std::cerr << "Erreur: NaN dans l'intégration" << std::endl;
        return Eigen::VectorXd::Zero(rows);
    }
    
    return result;
}

/**
 * Charge une matrice à partir d'un fichier CSV
 * @param filename Nom du fichier
 * @return Matrice chargée
 */
inline Eigen::MatrixXd loadCSV(const std::string &filename) {
    std::ifstream file(filename);
    if(!file) {
        std::cerr << "Erreur: Impossible d'ouvrir le fichier " << filename << std::endl;
        return Eigen::MatrixXd::Zero(0, 0);
    }
    
    std::vector<std::vector<double>> data;
    std::string line;
    while(std::getline(file, line)){
        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;
        while(std::getline(ss,val,',')){
            row.push_back(std::stod(val));
        }
        data.push_back(row);
    }
    
    if(data.empty()){
        std::cerr << "Erreur: Fichier vide " << filename << std::endl;
        return Eigen::MatrixXd::Zero(0, 0);
    }
    
    int rows = data.size();
    int cols = data[0].size();
    for(int i=0; i<rows; i++){
        if(static_cast<int>(data[i].size()) != cols){
            std::cerr << "Erreur: Les lignes du fichier " << filename << " n'ont pas la même longueur" << std::endl;
            return Eigen::MatrixXd::Zero(0, 0);
        }
    }
    
    Eigen::MatrixXd mat(rows,cols);
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            mat(i,j) = data[i][j];
        }
    }
    
    return mat;
}

} // namespace curve_utils

#endif // CURVE_UTILS_HPP