#ifndef VECTOR_FIELD_H
#define VECTOR_FIELD_H

#include "data_loader.h"
#include <Eigen/Dense>
#include <vector>

void compute_vector_field(const RunData& run_data, double w,
                          Eigen::ArrayXXd& Ux, Eigen::ArrayXXd& Uy,
                          Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& mask);

double compute_roughness(const Eigen::ArrayXXd& Ux, const Eigen::ArrayXXd& Uy, double w);

int compute_connections(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& mask);

#endif // VECTOR_FIELD_H
