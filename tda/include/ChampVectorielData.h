#ifndef CHAMP_VECTORIEL_DATA_H
#define CHAMP_VECTORIEL_DATA_H

#include <Eigen/Dense>

/*
 * Structure containing vector field data:
 * - Ux: X component of the vector field
 * - Uy: Y component of the vector field
 * - mask: Boolean mask indicating valid data points
 */
struct ChampVectorielData {
    Eigen::ArrayXXd Ux;      // X component of the vector field
    Eigen::ArrayXXd Uy;      // Y component of the vector field
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> mask;  // Validity mask
};

#endif // CHAMP_VECTORIEL_DATA_H
