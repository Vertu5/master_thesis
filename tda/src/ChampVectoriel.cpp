#include "ChampVectoriel.h"
#include <iostream>
#include <iomanip>

void print_matrix(const Eigen::MatrixXd& mat, const std::string& name, int max_rows, int max_cols) {
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

void print_array(const Eigen::VectorXd& vec, const std::string& name, int max_elements) {
    //std::cout << "Array: " << name << " (Showing up to " << max_elements << " elements)" << std::endl;
    int elements = std::min(static_cast<int>(vec.size()), max_elements);
    
    std::cout << std::fixed << std::setprecision(4);
    for(int i = 0; i < elements; ++i) {
        std::cout << vec(i) << " ";
    }
    if(vec.size() > max_elements) std::cout << "...";
    std::cout << std::endl;
}

void print_mask(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& mask, 
                const std::string& name, int max_rows, int max_cols) {
    //std::cout << "Mask: " << name << " (Showing up to " << max_rows << " rows and " << max_cols << " cols)" << std::endl;
    
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

Eigen::MatrixXd VectorFieldAnalyzer::concatenate_positions(
    const std::vector<RunData>& mission_data,
    bool is_x
) {
    size_t total_rows = 0;
    size_t max_cols = 0;
    for(const auto& run : mission_data) {
        total_rows += run.num_robots;
        max_cols = std::max(max_cols, run.num_time_steps);
    }
    
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(total_rows, max_cols);
    size_t current_row = 0;
    for(const auto& run : mission_data) {
        const auto& data = is_x ? run.x : run.y;
        for(size_t t = 0; t < run.num_time_steps; ++t) {
            for(size_t r = 0; r < run.num_robots; ++r) {
                result(current_row + r, t) = data[t][r];
            }
        }
        current_row += run.num_robots;
    }
    
    //print_matrix(result.topLeftCorner(5,5), is_x ? "all_x" : "all_y");
    return result;
}

Eigen::MatrixXd VectorFieldAnalyzer::concatenate_velocities(
    const std::vector<RunData>& mission_data,
    bool is_x
) {
    size_t total_rows = 0;
    size_t max_cols = 0;
    for(const auto& run : mission_data) {
        total_rows += run.num_robots;
        max_cols = std::max(max_cols, run.num_time_steps - 1);
    }
    
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(total_rows, max_cols);
    size_t current_row = 0;
    for(const auto& run : mission_data) {
        const auto& data = is_x ? run.vx : run.vy;
        for(size_t t = 0; t < run.num_time_steps - 1; ++t) {
            for(size_t r = 0; r < run.num_robots; ++r) {
                result(current_row + r, t) = data[t][r];
            }
        }
        current_row += run.num_robots;
    }
    
    //print_matrix(result.topLeftCorner(5,5), is_x ? "all_vx" : "all_vy");
    return result;
}

VectorFieldData VectorFieldAnalyzer::generate_vector_field(
    const Eigen::MatrixXd& x,
    const Eigen::MatrixXd& y,
    const Eigen::MatrixXd& vx,
    const Eigen::MatrixXd& vy,
    double w
) {
    // Transform coordinates
    Eigen::MatrixXd x_image = -y.array() + ARENA_SIZE/2;
    Eigen::MatrixXd y_image = -x.array() + ARENA_SIZE/2;
    Eigen::MatrixXd vx_image = -vy;
    Eigen::MatrixXd vy_image = -vx;

    //print_matrix(x_image.topLeftCorner(5,5), "x_image");
    //print_matrix(y_image.topLeftCorner(5,5), "y_image");
    //print_matrix(vx_image.topLeftCorner(5,5), "vx_image");
    //print_matrix(vy_image.topLeftCorner(5,5), "vy_image");

    // Calculate grid dimensions
    int nx_bins = int(2 * std::ceil(0.5 * ARENA_SIZE / w));
    int ny_bins = nx_bins;
    //std::cout << "Computed grid dimensions: nx_bins = " << nx_bins << ", ny_bins = " << ny_bins << std::endl;

    // Create bins
    double x_diff = nx_bins * w - ARENA_SIZE;
    double y_diff = ny_bins * w - ARENA_SIZE;
    Eigen::VectorXd x_bins = Eigen::VectorXd::LinSpaced(nx_bins + 1, -x_diff/2, ARENA_SIZE + x_diff/2);
    Eigen::VectorXd y_bins = Eigen::VectorXd::LinSpaced(ny_bins + 1, -y_diff/2, ARENA_SIZE + y_diff/2);

    //print_array(x_bins.head(14), "x_bins");
    //print_array(y_bins.head(14), "y_bins");

    // Initialize accumulation matrices
    Eigen::ArrayXXi M = Eigen::ArrayXXi::Zero(ny_bins, nx_bins);
    Eigen::ArrayXXd V_sum_x = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);
    Eigen::ArrayXXd V_sum_y = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);

    // Accumulate values
    for(int t = 0; t < vx_image.cols(); ++t) {
        for(int r = 0; r < vx_image.rows(); ++r) {
            // Find bin indices
            int x_idx = std::lower_bound(x_bins.data(), x_bins.data() + nx_bins + 1, 
                x_image(r,t)) - x_bins.data() - 1;
            int y_idx = std::lower_bound(y_bins.data(), y_bins.data() + ny_bins + 1, 
                y_image(r,t)) - y_bins.data() - 1;

            if(x_idx >= 0 && x_idx < nx_bins && y_idx >= 0 && y_idx < ny_bins) {
                M(y_idx, x_idx) += 1;
                V_sum_x(y_idx, x_idx) += vx_image(r,t);
                V_sum_y(y_idx, x_idx) += vy_image(r,t);
            }
        }
    }

    // Create result
    VectorFieldData result;
    result.Ux = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);
    result.Uy = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);
    result.mask = M.array() > 0;

    // Calculate final vector field
    for(int i = 0; i < ny_bins; ++i) {
        for(int j = 0; j < nx_bins; ++j) {
            if(result.mask(i,j)) {
                result.Ux(i,j) = V_sum_x(i,j) / M(i,j);
                result.Uy(i,j) = V_sum_y(i,j) / M(i,j);
            }
        }
    }

    // Print results
    //print_matrix(result.Ux.topLeftCorner(5,5), "result.Ux");
    //print_matrix(result.Uy.topLeftCorner(5,5), "result.Uy");
    //print_mask(result.mask, "result.mask", result.mask.rows(), result.mask.cols());

    return result;
}

VectorFieldData VectorFieldAnalyzer::generate_mission_vector_field(
    const std::vector<RunData>& mission_data,
    double bin_width
) {
    Eigen::MatrixXd all_x = concatenate_positions(mission_data, true);
    Eigen::MatrixXd all_y = concatenate_positions(mission_data, false);
    Eigen::MatrixXd all_vx = concatenate_velocities(mission_data, true);
    Eigen::MatrixXd all_vy = concatenate_velocities(mission_data, false);
    
    return generate_vector_field(all_x, all_y, all_vx, all_vy, bin_width);
}