#include "vector_field.h"
#include <cmath>
#include <iostream>

void compute_vector_field(const RunData& run_data, double w,
                          Eigen::ArrayXXd& Ux, Eigen::ArrayXXd& Uy,
                          Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& mask) {
    const double R_ARENA_SIZE = 2 * 1.231;
    size_t num_time_steps = run_data.num_time_steps - 1; // Adjust for velocity time steps
    size_t num_robots = run_data.num_robots;

    // Transform positions to image coordinates
    Eigen::ArrayXXd x_image(num_time_steps, num_robots);
    Eigen::ArrayXXd y_image(num_time_steps, num_robots);
    for (size_t t = 0; t < num_time_steps; ++t) {
        for (size_t i = 0; i < num_robots; ++i) {
            x_image(t, i) = -run_data.y[t][i] + R_ARENA_SIZE / 2;
            y_image(t, i) = -run_data.x[t][i] + R_ARENA_SIZE / 2;
        }
    }

    // Velocities in image coordinates
    Eigen::ArrayXXd vx_image(num_time_steps, num_robots);
    Eigen::ArrayXXd vy_image(num_time_steps, num_robots);
    for (size_t t = 0; t < num_time_steps; ++t) {
        for (size_t i = 0; i < num_robots; ++i) {
            vx_image(t, i) = -run_data.vy[t][i];
            vy_image(t, i) = -run_data.vx[t][i];
        }
    }

    // Grid dimensions
    int nx_bins = static_cast<int>(2 * std::ceil(0.5 * R_ARENA_SIZE / w));
    int ny_bins = nx_bins;

    // Grid edges
    double x_diff = nx_bins * w - R_ARENA_SIZE;
    double y_diff = ny_bins * w - R_ARENA_SIZE;
    Eigen::VectorXd x_bins = Eigen::VectorXd::LinSpaced(nx_bins + 1, -x_diff / 2, R_ARENA_SIZE + x_diff / 2);
    Eigen::VectorXd y_bins = Eigen::VectorXd::LinSpaced(ny_bins + 1, -y_diff / 2, R_ARENA_SIZE + y_diff / 2);

    // Initialize accumulators
    Eigen::ArrayXXi M = Eigen::ArrayXXi::Zero(ny_bins, nx_bins);
    Eigen::ArrayXXd Vx_sum = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);
    Eigen::ArrayXXd Vy_sum = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);

    // Accumulate positions and velocities
    for (size_t t = 0; t < num_time_steps; ++t) {
        for (size_t i = 0; i < num_robots; ++i) {
            double xi = x_image(t, i);
            double yi = y_image(t, i);

            int x_idx = std::lower_bound(x_bins.data(), x_bins.data() + x_bins.size(), xi) - x_bins.data() - 1;
            int y_idx = std::lower_bound(y_bins.data(), y_bins.data() + y_bins.size(), yi) - y_bins.data() - 1;

            if (x_idx >= 0 && x_idx < nx_bins && y_idx >= 0 && y_idx < ny_bins) {
                M(y_idx, x_idx) += 1;
                Vx_sum(y_idx, x_idx) += vx_image(t, i);
                Vy_sum(y_idx, x_idx) += vy_image(t, i);
            }
        }
    }

    // Compute Ux and Uy
    Ux = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);
    Uy = Eigen::ArrayXXd::Zero(ny_bins, nx_bins);
    mask = M > 0;

    for (int i = 0; i < ny_bins; ++i) {
        for (int j = 0; j < nx_bins; ++j) {
            if (M(i, j) > 0) {
                Ux(i, j) = Vx_sum(i, j) / M(i, j);
                Uy(i, j) = Vy_sum(i, j) / M(i, j);
            }
        }
    }
}

double compute_roughness(const Eigen::ArrayXXd& Ux, const Eigen::ArrayXXd& Uy, double w) {
    int ny = Ux.rows();
    int nx = Ux.cols();
    double w2 = w * w;

    // Compute second derivatives
    Eigen::ArrayXXd Ux_xx = Eigen::ArrayXXd::Zero(ny, nx);
    Eigen::ArrayXXd Ux_yy = Eigen::ArrayXXd::Zero(ny, nx);
    Eigen::ArrayXXd Ux_xy = Eigen::ArrayXXd::Zero(ny, nx);
    Eigen::ArrayXXd Uy_xx = Eigen::ArrayXXd::Zero(ny, nx);
    Eigen::ArrayXXd Uy_yy = Eigen::ArrayXXd::Zero(ny, nx);
    Eigen::ArrayXXd Uy_xy = Eigen::ArrayXXd::Zero(ny, nx);

    // Compute Ux_xx and Uy_xx
    for (int i = 0; i < ny; ++i) {
        for (int j = 1; j < nx - 1; ++j) {
            Ux_xx(i, j) = (Ux(i, j + 1) - 2 * Ux(i, j) + Ux(i, j - 1)) / w2;
            Uy_xx(i, j) = (Uy(i, j + 1) - 2 * Uy(i, j) + Uy(i, j - 1)) / w2;
        }
    }

    // Compute Ux_yy and Uy_yy
    for (int i = 1; i < ny - 1; ++i) {
        for (int j = 0; j < nx; ++j) {
            Ux_yy(i, j) = (Ux(i + 1, j) - 2 * Ux(i, j) + Ux(i - 1, j)) / w2;
            Uy_yy(i, j) = (Uy(i + 1, j) - 2 * Uy(i, j) + Uy(i - 1, j)) / w2;
        }
    }

    // Compute Ux_xy and Uy_xy
    for (int i = 1; i < ny - 1; ++i) {
        for (int j = 1; j < nx - 1; ++j) {
            Ux_xy(i, j) = (Ux(i + 1, j + 1) - Ux(i + 1, j - 1) - Ux(i - 1, j + 1) + Ux(i - 1, j - 1)) / (4 * w2);
            Uy_xy(i, j) = (Uy(i + 1, j + 1) - Uy(i + 1, j - 1) - Uy(i - 1, j + 1) + Uy(i - 1, j - 1)) / (4 * w2);
        }
    }

    // Compute norms
    Eigen::ArrayXXd norm_Hx_squared = Ux_xx.square() + 2 * Ux_xy.square() + Ux_yy.square();
    Eigen::ArrayXXd norm_Hy_squared = Uy_xx.square() + 2 * Uy_xy.square() + Uy_yy.square();

    // Compute roughness
    double R_Ux = norm_Hx_squared.sum() * w2;
    double R_Uy = norm_Hy_squared.sum() * w2;
    double R = R_Ux + R_Uy;

    return R * w * w;
}

int compute_connections(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& mask) {
    int ny = mask.rows();
    int nx = mask.cols();
    int total_connections = 0;

    // Define 8-neighbourhood kernel
    int dx[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
    int dy[8] = {-1, 0, 1, 1, 1, 0, -1, -1};

    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (mask(i, j)) {
                for (int k = 0; k < 8; ++k) {
                    int ni = i + dy[k];
                    int nj = j + dx[k];
                    if (ni >= 0 && ni < ny && nj >= 0 && nj < nx && mask(ni, nj)) {
                        total_connections++;
                    }
                }
            }
        }
    }

    // Each connection counted twice
    total_connections /= 2;
    return total_connections;
}
