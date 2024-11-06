#ifndef HHD_ANALYSIS_H
#define HHD_ANALYSIS_H

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <iomanip>
#include "vector_field.h"

struct HHDComponents {
    VectorFieldData divergent;    // nD component
    VectorFieldData rotational;   // nRu component
    double mean_divergent;
    double mean_rotational;
};

class HHDAnalyzer {
private:
    static Eigen::MatrixXd solvePoissonEquation(const Eigen::MatrixXd& f, const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& mask, 
                                               double optimal_w) {
        int rows = f.rows();
        int cols = f.cols();
        int n = rows * cols;
        
        Eigen::SparseMatrix<double> A(n, n);
        Eigen::VectorXd b(n), x(n);
        std::vector<Eigen::Triplet<double>> triplets;
        
        double w2 = optimal_w * optimal_w;
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                
                if(mask(i, j)) {
                    triplets.push_back(Eigen::Triplet<double>(idx, idx, -4.0/w2));
                    
                    if(i > 0 && mask(i-1, j)) 
                        triplets.push_back(Eigen::Triplet<double>(idx, (i-1)*cols + j, 1.0/w2));
                    if(i < rows-1 && mask(i+1, j))
                        triplets.push_back(Eigen::Triplet<double>(idx, (i+1)*cols + j, 1.0/w2));
                    if(j > 0 && mask(i, j-1))
                        triplets.push_back(Eigen::Triplet<double>(idx, i*cols + (j-1), 1.0/w2));
                    if(j < cols-1 && mask(i, j+1))
                        triplets.push_back(Eigen::Triplet<double>(idx, i*cols + (j+1), 1.0/w2));
                    
                    b(idx) = f(i, j);
                } else {
                    triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                    b(idx) = 0.0;
                }
            }
        }
        
        A.setFromTriplets(triplets.begin(), triplets.end());
        
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
        solver.setMaxIterations(1000);
        solver.setTolerance(1e-6);
        x = solver.compute(A).solve(b);
        
        Eigen::MatrixXd result(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result(i, j) = x(i * cols + j);
            }
        }
        
        return result;
    }

    static double calculate_component_mean(const VectorFieldData& field) {
        double sum = 0.0;
        int count = 0;
        
        for(int i = 0; i < field.Ux.rows(); i++) {
            for(int j = 0; j < field.Ux.cols(); j++) {
                if(field.mask(i, j)) {
                    double magnitude = std::sqrt(
                        field.Ux(i, j) * field.Ux(i, j) + 
                        field.Uy(i, j) * field.Uy(i, j)
                    );
                    sum += magnitude;
                    count++;
                }
            }
        }
        
        return count > 0 ? sum / count : 0.0;
    }

public:
    static std::pair<std::vector<VectorFieldData>, std::vector<HHDComponents>> 
    analyze_missions(const std::vector<std::string>& mission_names,
                    const std::vector<std::vector<RunData>>& all_mission_data,
                    const std::vector<VectorFieldData>& vector_fields,
                    double optimal_w) {
        
        std::vector<HHDComponents> all_components;
        std::cout << std::fixed << std::setprecision(6);
        
        for(size_t idx = 0; idx < vector_fields.size(); idx++) {
            const auto& field = vector_fields[idx];
            
            // Compute divergence
            Eigen::MatrixXd div = Eigen::MatrixXd::Zero(field.Ux.rows(), field.Ux.cols());
            for(int i = 1; i < field.Ux.rows()-1; i++) {
                for(int j = 1; j < field.Ux.cols()-1; j++) {
                    if(field.mask(i, j)) {
                        div(i, j) = (field.Ux(i, j+1) - field.Ux(i, j-1))/(2*optimal_w) +
                                   (field.Uy(i+1, j) - field.Uy(i-1, j))/(2*optimal_w);
                    }
                }
            }
            
            // Compute curl
            Eigen::MatrixXd curl = Eigen::MatrixXd::Zero(field.Ux.rows(), field.Ux.cols());
            for(int i = 1; i < field.Ux.rows()-1; i++) {
                for(int j = 1; j < field.Ux.cols()-1; j++) {
                    if(field.mask(i, j)) {
                        curl(i, j) = (field.Uy(i, j+1) - field.Uy(i, j-1))/(2*optimal_w) -
                                    (field.Ux(i+1, j) - field.Ux(i-1, j))/(2*optimal_w);
                    }
                }
            }
            
            // Solve Poisson equations
            Eigen::MatrixXd phi = solvePoissonEquation(div, field.mask, optimal_w);
            Eigen::MatrixXd psi = solvePoissonEquation(curl, field.mask, optimal_w);
            
            // Initialize components
            HHDComponents components;
            components.divergent.Ux = Eigen::MatrixXd::Zero(field.Ux.rows(), field.Ux.cols());
            components.divergent.Uy = Eigen::MatrixXd::Zero(field.Ux.rows(), field.Ux.cols());
            components.divergent.mask = field.mask;  // Using original boolean mask
            
            components.rotational.Ux = Eigen::MatrixXd::Zero(field.Ux.rows(), field.Ux.cols());
            components.rotational.Uy = Eigen::MatrixXd::Zero(field.Ux.rows(), field.Ux.cols());
            components.rotational.mask = field.mask;  // Using original boolean mask
            
            // Compute components
            for(int i = 1; i < field.Ux.rows()-1; i++) {
                for(int j = 1; j < field.Ux.cols()-1; j++) {
                    if(field.mask(i, j)) {
                        components.divergent.Ux(i, j) = (phi(i, j+1) - phi(i, j-1))/(2*optimal_w);
                        components.divergent.Uy(i, j) = (phi(i+1, j) - phi(i-1, j))/(2*optimal_w);
                        
                        components.rotational.Ux(i, j) = -(psi(i+1, j) - psi(i-1, j))/(2*optimal_w);
                        components.rotational.Uy(i, j) = (psi(i, j+1) - psi(i, j-1))/(2*optimal_w);
                    }
                }
            }
            
            // Calculate means
            components.mean_divergent = calculate_component_mean(components.divergent);
            components.mean_rotational = calculate_component_mean(components.rotational);
            
            // Print statistics
            std::cout << "Mission: " << mission_names[idx] << std::endl;
            std::cout << "Mean of nD: " << components.mean_divergent << std::endl;
            std::cout << "Mean of nRu: " << components.mean_rotational << std::endl;
            
            all_components.push_back(components);
        }
        
        return {vector_fields, all_components};
    }
};

#endif // HHD_ANALYSIS_H