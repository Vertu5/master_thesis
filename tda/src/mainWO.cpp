// main_save_extended_persistence.cpp
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include "data_loader.h"
#include "RGrid.h"
#include "VectorField.h"
#include "HHD.h"
#include "MaskedArray.h"
#include "GreensFunction.h"
#include "ExtendedPersistence.h"
#include "ChampVectorielGenerateur.h"

#include <chrono>

// Timing utilities
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;

public:
    Timer(const std::string& task_name) : name(task_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        std::cout << name << " took " << duration << " ms" << std::endl;
    }
};

// Structure to store timing statistics
struct TimingStats {
    double vector_field_time = 0.0;
    double divcurl_time = 0.0;
    double nhhd_time = 0.0;
    double persistence_time = 0.0;
    int num_missions = 0;

    void print_average() {
        if(num_missions == 0) return;
        std::cout << "\nAverage Timing Statistics:" << std::endl;
        std::cout << "=========================" << std::endl;
        std::cout << "Vector Field Generation: " << vector_field_time/num_missions << " ms" << std::endl;
        std::cout << "Divergence/Curl Calculation: " << divcurl_time/num_missions << " ms" << std::endl;
        std::cout << "Natural HHD: " << nhhd_time/num_missions << " ms" << std::endl;
        std::cout << "Persistence Calculation: " << persistence_time/num_missions << " ms" << std::endl;
    }
};

int main() {
    const double optimal_w = 0.2043;
    const double R_ARENA_SIZE = 2 * 1.231;
    TimingStats timing_stats;

    std::cout << "Optimal bin width: " << optimal_w << std::endl;
    std::cout << "Arena size: " << R_ARENA_SIZE << std::endl;

    Timer total_timer("Total Execution");

    // Data loading
    Timer loading_timer("Data Loading");
    DataLoader data_loader(".");
    auto [all_mission_data, missions] = data_loader.loadAllMissions();

    if (all_mission_data.empty()) {
        std::cerr << "No mission data loaded." << std::endl;
        return 1;
    }

    // Process each mission
    for (size_t mission_idx = 0; mission_idx < all_mission_data.size(); ++mission_idx) {
        const auto& mission_data = all_mission_data[mission_idx];
        std::string mission_name = missions[mission_idx];
        std::cout << "\nProcessing Mission " << mission_name << std::endl;

        // Grid setup
        Timer grid_timer("Grid Setup");
        size_t nx_bins = static_cast<size_t>(2 * std::ceil(0.5 * R_ARENA_SIZE / optimal_w));
        size_t ny_bins = nx_bins;
        RGrid rgrid(nx_bins, ny_bins, optimal_w, optimal_w);

        // Generate both fields
        Timer vf_timer("Field Generation");
        auto fields = ChampVectorielGenerateur::generate_mission_fields(mission_data, optimal_w);
        
        // Create VectorField for velocity
        VectorField<double> vfield(std::vector<size_t>{ny_bins, nx_bins});
        for (size_t i = 0; i < ny_bins; ++i) {
            for (size_t j = 0; j < nx_bins; ++j) {
                size_t idx = i * nx_bins + j;
                vfield.u.data[idx] = fields.velocity.Ux(i, j);
                vfield.v.data[idx] = fields.velocity.Uy(i, j);
                vfield.u.mask[idx] = !fields.mask(i, j);
                vfield.v.mask[idx] = !fields.mask(i, j);
            }
        }

        // Process velocity field (HHD + Persistence)
        {
            // Divergence and curl calculation
            auto start = std::chrono::high_resolution_clock::now();
            vfield.need_divcurl(rgrid);
            auto end = std::chrono::high_resolution_clock::now();
            timing_stats.divcurl_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count();

            // Natural HHD
            start = std::chrono::high_resolution_clock::now();
            std::vector<MaskedArray<double>> input_fields = {vfield.div, vfield.curl};
            naturalHHD<double> nhhd(input_fields, rgrid);
            end = std::chrono::high_resolution_clock::now();
            timing_stats.nhhd_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count();

            // Persistence calculation for both O and velocity fields
            start = std::chrono::high_resolution_clock::now();
            
            // Setup for HHD persistence
            MaskedArray<bool> mask_D(nhhd.D.shape);
            mask_D.mask = nhhd.D.mask;
            MaskedArray<bool> mask_Ru(nhhd.Ru.shape);
            mask_Ru.mask = nhhd.Ru.mask;

            // Setup for density field persistence
            MaskedArray<double> density_field(std::vector<size_t>{ny_bins, nx_bins});
            MaskedArray<bool> density_mask(std::vector<size_t>{ny_bins, nx_bins});
            
            // Copy density field to MaskedArray format
            for(size_t i = 0; i < ny_bins; ++i) {
                for(size_t j = 0; j < nx_bins; ++j) {
                    density_field.data[i * nx_bins + j] = fields.O(i, j);
                    density_mask.mask[i * nx_bins + j] = !fields.mask(i, j);
                }
            }

            std::array<double, 2> dx = {optimal_w, optimal_w};
            ExtendedPersistenceCalculator calculator;

            // Compute persistence for velocity field (12 diagrams)
            auto divergent_result = calculator.computeExtendedPersistence(
                nhhd.D, mask_D, dx, nx_bins, ny_bins);
            auto rotational_result = calculator.computeExtendedPersistence(
                -nhhd.Ru, mask_Ru, dx, nx_bins, ny_bins);
                
            // Compute persistence for density field (6 diagrams)
            auto density_result = calculator.computeExtendedPersistence(
                density_field, density_mask, dx, nx_bins, ny_bins);

            end = std::chrono::high_resolution_clock::now();
            timing_stats.persistence_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count();

            // Save all persistence diagrams (18 in total)
            std::string filename = mission_name + "_extended_persistence.dat";
            std::ofstream outfile(filename, std::ios::binary);
            if (!outfile.is_open()) {
                std::cerr << "Failed to open file for writing: " << filename << std::endl;
                continue;
            }

            // Collect all diagrams (18 in total)
            std::vector<PersistenceDiagram> diagrams = {
                // Velocity field diagrams (12)
                divergent_result.ord_h0, divergent_result.ord_h1,
                divergent_result.rel_h1, divergent_result.rel_h2,
                divergent_result.ext_plus_h0, divergent_result.ext_minus_h1,
                rotational_result.ord_h0, rotational_result.ord_h1,
                rotational_result.rel_h1, rotational_result.rel_h2,
                rotational_result.ext_plus_h0, rotational_result.ext_minus_h1,
                // Density field diagrams (6)
                density_result.ord_h0, density_result.ord_h1,
                density_result.rel_h1, density_result.rel_h2,
                density_result.ext_plus_h0, density_result.ext_minus_h1
            };

            // Write all diagrams
            for (const auto& diagram : diagrams) {
                size_t num_pairs = diagram.size();
                outfile.write(reinterpret_cast<const char*>(&num_pairs), sizeof(size_t));
                for (const auto& pair : diagram) {
                    double birth = pair.first;
                    double death = pair.second;
                    outfile.write(reinterpret_cast<const char*>(&birth), sizeof(double));
                    outfile.write(reinterpret_cast<const char*>(&death), sizeof(double));
                }
            }
            outfile.close();
            std::cout << "Extended persistence diagrams saved to " << filename 
                     << " (18 diagrams: 12 for velocity field, 6 for density field)" << std::endl;
        }
        timing_stats.num_missions++;
    }

    // Print timing statistics
    timing_stats.print_average();
    return 0;
}