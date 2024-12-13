#ifndef CHAMP_VECTORIEL_UTILS_H
#define CHAMP_VECTORIEL_UTILS_H

#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <iomanip>

namespace ChampVectorielUtils {
    inline void print_matrix(const Eigen::MatrixXd& mat, const std::string& name, int max_rows = 5, int max_cols = 5) {
        std::cout << "Matrix: " << name << " (Showing up to " << max_rows << " rows and " << max_cols << " cols)" << std::endl;
        int rows = std::min(static_cast<int>(mat.rows()), max_rows);
        int cols = std::min(static_cast<int>(mat.cols()), max_cols);
        
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                std::cout << std::fixed << std::setprecision(4) << mat(i,j) << " ";
            }
            std::cout << (mat.cols() > max_cols ? "..." : "") << std::endl;
        }
        if(mat.rows() > max_rows) std::cout << "..." << std::endl;
    }

    inline void print_array(const Eigen::VectorXd& vec, const std::string& name, int max_elements = 10) {
        int elements = std::min(static_cast<int>(vec.size()), max_elements);
        
        std::cout << std::fixed << std::setprecision(4);
        for(int i = 0; i < elements; ++i) {
            std::cout << vec(i) << " ";
        }
        if(vec.size() > max_elements) std::cout << "...";
        std::cout << std::endl;
    }

    inline void print_mask(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& mask, 
                    const std::string& name, int max_rows = 10, int max_cols = 10) {
        int rows = std::min(static_cast<int>(mask.rows()), max_rows);
        int cols = std::min(static_cast<int>(mask.cols()), max_cols);
        
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                std::cout << (mask(i,j) ? "1 " : "0 ");
            }
            if(mask.cols() > max_cols) std::cout << "...";
            std::cout << std::endl;
        }
        if(mask.rows() > max_rows) std::cout << "..." << std::endl;
    }
}

#endif // CHAMP_VECTORIEL_UTILS_H