// HelmholtzHodgeDecomposition.h
#ifndef HELMHOLTZ_HODGE_DECOMPOSITION_H
#define HELMHOLTZ_HODGE_DECOMPOSITION_H

#pragma once
#include <Eigen/Dense>
#include "RGrid.h"
#include "VectorField.h"
#include "HHD.h"

class HHDAnalyzer {
public:
    HHDAnalyzer(size_t ny, size_t nx, double dx) 
        : m_grid(nx, ny, dx, dx) {
        m_vfield = std::make_unique<VectorField<float>>();
        m_vfield->dim = 2;
        m_vfield->sz = nx * ny;
    }

    void calculate_hhd(const Eigen::MatrixXd& Ux, const Eigen::MatrixXd& Uy,
                      const Eigen::Matrix<bool, -1, -1>& mask);

    std::pair<double, double> get_means() const;
    
private:
    RGrid m_grid;
    std::unique_ptr<VectorField<float>> m_vfield;
    naturalHHD<float>* m_hhd = nullptr;
};
#endif

