#ifndef CHAMPVECTORIEL_H
#define CHAMPVECTORIEL_H

#include <Eigen/Dense>
#include "data_loader.h"

struct VectorFieldData {
    Eigen::ArrayXXd Ux;
    Eigen::ArrayXXd Uy;
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> mask;
};

void print_matrix(const Eigen::MatrixXd& mat, const std::string& name, int max_rows = 5, int max_cols = 5);
void print_array(const Eigen::VectorXd& vec, const std::string& name, int max_elements = 10);
void print_mask(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& mask, const std::string& name, int max_rows = 10, int max_cols = 10);

class VectorFieldAnalyzer {
public:
    static constexpr double ARENA_SIZE = 2 * 1.231;

    static VectorFieldData generate_mission_vector_field(
        const std::vector<RunData>& mission_data,
        double bin_width
    );

private:
    static VectorFieldData generate_vector_field(
        const Eigen::MatrixXd& x,
        const Eigen::MatrixXd& y,
        const Eigen::MatrixXd& vx,
        const Eigen::MatrixXd& vy,
        double w
    );

    static Eigen::MatrixXd concatenate_positions(
        const std::vector<RunData>& mission_data,
        bool is_x
    );

    static Eigen::MatrixXd concatenate_velocities(
        const std::vector<RunData>& mission_data,
        bool is_x
    );
};

#endif // CHAMPVECTORIEL_H