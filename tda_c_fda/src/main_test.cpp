#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include "RGrid.h"
#include "VectorField.h"
#include "HHD.h"
#include "MaskedArray.h"
#include "GreensFunction.h"
#include "ExtendedPersistence.h"

struct ChampVectorielData {
    Eigen::ArrayXXd Ux;
    Eigen::ArrayXXd Uy;
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> mask;
};

void print_mask_visualization(const ChampVectorielData& field) {
    if(field.mask.size() == 0) {
        std::cout << "Mask is empty!" << std::endl;
        return;
    }
    std::cout << "\nMask Visualization (# = masked (true), . = valid (false)):" << std::endl;
    for(int i = 0; i < field.mask.rows(); ++i) {
        for(int j = 0; j < field.mask.cols(); ++j) {
            std::cout << (field.mask(i,j) ? "#" : ".") << " ";
        }
        std::cout << std::endl;
    }
}

ChampVectorielData create_simple_convergence(size_t size) {
    ChampVectorielData field;
    // Initialisation explicite des dimensions
    field.Ux = Eigen::ArrayXXd::Zero(size, size);
    field.Uy = Eigen::ArrayXXd::Zero(size, size);
    field.mask = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(size, size, false);
    
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            double x = -1.0 + 2.0 * j / (size - 1);
            double y = -1.0 + 2.0 * i / (size - 1);
            
            double dx = -x;
            double dy = -y;
            
            double magnitude = std::sqrt(dx*dx + dy*dy);
            if (magnitude < 1e-10) magnitude = 1e-10;
            
            field.Ux(i, j) = dx / magnitude;
            field.Uy(i, j) = dy / magnitude;
        }
    }
    
    // Masquer uniquement les bords
    for(size_t j = 0; j < size; ++j) {
        field.mask(0, j) = true;         // Première ligne
        field.mask(size-1, j) = true;    // Dernière ligne
        field.mask(j, 0) = true;         // Première colonne
        field.mask(j, size-1) = true;    // Dernière colonne
    }
    
    return field;
}

ChampVectorielData create_damped_convergence(size_t size) {
    ChampVectorielData field;
    // Initialisation explicite des dimensions
    field.Ux = Eigen::ArrayXXd::Zero(size, size);
    field.Uy = Eigen::ArrayXXd::Zero(size, size);
    field.mask = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(size, size, false);
    
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            double x = -1.0 + 2.0 * j / (size - 1);
            double y = -1.0 + 2.0 * i / (size - 1);
            
            double dx = -x;
            double dy = -y;
            
            double magnitude = std::sqrt(dx*dx + dy*dy);
            if (magnitude < 1e-10) magnitude = 1e-10;
            
            double damping = std::sqrt(x*x + y*y);
            field.Ux(i, j) = (dx / magnitude) * damping;
            field.Uy(i, j) = (dy / magnitude) * damping;
        }
    }
    
    // Masquer uniquement les bords
    for(size_t j = 0; j < size; ++j) {
        field.mask(0, j) = true;
        field.mask(size-1, j) = true;
        field.mask(j, 0) = true;
        field.mask(j, size-1) = true;
    }
    
    return field;
}

VectorField<double> convert_to_vectorfield(const ChampVectorielData& field) {
    size_t size = field.Ux.rows();
    VectorField<double> vfield(std::vector<size_t>{size, size});
    
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            size_t idx = i * size + j;
            vfield.u.data[idx] = field.Ux(i, j);
            vfield.v.data[idx] = field.Uy(i, j);
            vfield.u.mask[idx] = field.mask(i, j);
            vfield.v.mask[idx] = field.mask(i, j);
        }
    }
    
    return vfield;
}

int main() {
    try {
        const double optimal_w = 0.2043;
        const size_t size = 14;
        
        std::vector<std::string> mission_names = {"mission_test_1", "mission_test_2"};
        
        // Créer les deux champs vectoriels
        ChampVectorielData field1 = create_simple_convergence(size);
        ChampVectorielData field2 = create_damped_convergence(size);
        
        std::cout << "Simple Convergence Field Mask:" << std::endl;
        print_mask_visualization(field1);
        std::cout << "\nDamped Convergence Field Mask:" << std::endl;
        print_mask_visualization(field2);
        
        std::vector<ChampVectorielData> fields = {field1, field2};
        
        // Traiter chaque champ
        for(size_t field_idx = 0; field_idx < fields.size(); ++field_idx) {
            RGrid rgrid(size, size, optimal_w, optimal_w);
            VectorField<double> vfield = convert_to_vectorfield(fields[field_idx]);
            
            vfield.need_divcurl(rgrid);
            std::vector<MaskedArray<double>> input_fields = {vfield.div, vfield.curl};
            naturalHHD<double> nhhd(input_fields, rgrid);
            
            MaskedArray<bool> mask_D(nhhd.D.shape);
            mask_D.mask = nhhd.D.mask;
            MaskedArray<bool> mask_Ru(nhhd.Ru.shape);
            mask_Ru.mask = nhhd.Ru.mask;
            
            std::array<double, 2> dx = {optimal_w, optimal_w};
            ExtendedPersistenceCalculator calculator;
            
            auto divergent_result = calculator.computeExtendedPersistence(
                nhhd.D, mask_D, dx, size, size);
            auto rotational_result = calculator.computeExtendedPersistence(
                -nhhd.Ru, mask_Ru, dx, size, size);
            
            std::string filename = mission_names[field_idx] + "_extended_persistence.dat";
            std::ofstream outfile(filename, std::ios::binary);
            
            if (!outfile.is_open()) {
                throw std::runtime_error("Failed to open output file: " + filename);
            }
            
            std::vector<PersistenceDiagram> diagrams = {
                divergent_result.ord_h0, divergent_result.ord_h1,
                divergent_result.rel_h1, divergent_result.rel_h2,
                divergent_result.ext_plus_h0, divergent_result.ext_minus_h1,
                rotational_result.ord_h0, rotational_result.ord_h1,
                rotational_result.rel_h1, rotational_result.rel_h2,
                rotational_result.ext_plus_h0, rotational_result.ext_minus_h1
            };
            
            for (const auto& diagram : diagrams) {
                size_t num_pairs = diagram.size();
                outfile.write(reinterpret_cast<const char*>(&num_pairs), sizeof(size_t));
                for (const auto& pair : diagram) {
                    outfile.write(reinterpret_cast<const char*>(&pair.first), sizeof(double));
                    outfile.write(reinterpret_cast<const char*>(&pair.second), sizeof(double));
                }
            }
            
            outfile.close();
            std::cout << "Saved " << filename << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
