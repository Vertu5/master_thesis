// main.cpp
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "data_loader.h"
#include "RGrid.h"
#include "VectorField.h"
#include "HHD.h"
#include "MaskedArray.h"
#include "GreensFunction.h"
#include "extended_persistence.h"
#include "ChampVectoriel.h"
#include <chrono>

template <typename T>
T calculate_mean_unmasked(const MaskedArray<T>& field) {
    T sum = 0;
    size_t count = 0;

    for (size_t i = 0; i < field.data.size(); ++i) {
        if (!field.mask[i]) {
            sum += std::abs(field.data[i]);
            count++;
        }
    }

    return count > 0 ? sum / count : 0;
}

template <typename T>
void print_field(const MaskedArray<T>& field, const std::string& name, size_t max_nx, size_t max_ny) {
    std::cout << "\n" << name << " (Limité à " << max_ny << "x" << max_nx << "):" << std::endl;
    size_t ny = std::min(static_cast<size_t>(14), max_ny);
    size_t nx = std::min(static_cast<size_t>(14), max_nx);

    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            size_t idx = y * max_nx + x;
            if (field.mask[idx]) {
                std::cout << "   ---   ";
            } else {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << field.data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
}

void print_persistence_diagrams(const std::string& mission, 
                                const ExtendedPersistenceCalculator::ExtendedPersistenceResult& result) {
    std::cout << "\nPersistence Diagrams - Mission " << mission << std::endl;

    const std::vector<std::string> titles = {"Ord H0", "Ord H1", "Rel H1", "Rel H2", "Ext+ H0", "Ext- H1"};
    const std::vector<std::pair<std::string, PersistenceDiagram>> diagrams = {
        {"Ordinary H0", result.ord_h0}, {"Ordinary H1", result.ord_h1},
        {"Relative H1", result.rel_h1}, {"Relative H2", result.rel_h2},
        {"Extended+ H0", result.ext_plus_h0}, {"Extended- H1", result.ext_minus_h1}
    };

    for (const auto& [name, diagram] : diagrams) {
        std::cout << "\n" << name << ":" << std::endl;
        if (diagram.empty()) {
            std::cout << "No persistence pairs" << std::endl;
        } else {
            for (const auto& pair : diagram) {
                std::cout << "Birth: " << std::fixed << std::setprecision(4) 
                          << pair.first << ", Death: " << pair.second << std::endl;
            }
        }
    }
}

class Timer {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    TimePoint start;
public:
    Timer() : start(Clock::now()) {}
    double elapsed() {
        auto end = Clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
};

struct TimingResults {
    double data_loading = 0;
    double vector_field = 0;
    double div_curl = 0;
    double nhhd = 0;
    double persistence_div = 0;
    double persistence_rot = 0;
};

int main() {
    const double optimal_w = 0.2043;
    const double R_ARENA_SIZE = 2 * 1.231;
    TimingResults timing;

    // Data loading
    Timer t1;
    DataLoader data_loader(".");
    auto all_mission_data = data_loader.loadAllMissions();
    timing.data_loading = t1.elapsed();

    if (all_mission_data.empty()) {
        std::cerr << "No mission data loaded." << std::endl;
        return 1;
    }

    //#pragma omp parallel for schedule(dynamic)
    for (size_t mission_idx = 0; mission_idx < all_mission_data.size(); ++mission_idx) {
        const auto& mission_data = all_mission_data[mission_idx];
        std::string mission = std::to_string(mission_idx + 1);

        // Grid setup
        size_t nx_bins = static_cast<size_t>(2 * std::ceil(0.5 * R_ARENA_SIZE / optimal_w));
        size_t ny_bins = nx_bins;
        RGrid rgrid(nx_bins, ny_bins, optimal_w, optimal_w);

        // Vector field generation
        Timer t2;
        VectorField<double> vfield(std::vector<size_t>{ny_bins, nx_bins});
        VectorFieldAnalyzer analyzer;
        auto field_data = analyzer.generate_mission_vector_field(mission_data, optimal_w);

        // Parallel field filling
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < ny_bins; ++i) {
            for (size_t j = 0; j < nx_bins; ++j) {
                size_t idx = i * nx_bins + j;
                vfield.u.data[idx] = field_data.Ux(i, j);
                vfield.v.data[idx] = field_data.Uy(i, j);
                vfield.u.mask[idx] = !field_data.mask(i, j);
                vfield.v.mask[idx] = !field_data.mask(i, j);
            }
        }
        #pragma omp atomic
        timing.vector_field += t2.elapsed();

        // Divergence and curl calculation
        Timer t3;
        vfield.need_divcurl(rgrid);
        #pragma omp atomic
        timing.div_curl += t3.elapsed();

        // Natural HHD computation
        Timer t4;
        std::vector<MaskedArray<double>> input_fields = {vfield.div, vfield.curl};
        naturalHHD<double> nhhd(input_fields, rgrid);
        #pragma omp atomic
        timing.nhhd += t4.elapsed();

        // Prepare masks
        MaskedArray<bool> mask_D(nhhd.D.shape);
        mask_D.mask = nhhd.D.mask;

        MaskedArray<bool> mask_Ru(nhhd.Ru.shape);
        mask_Ru.mask = nhhd.Ru.mask;

        std::array<double, 2> dx = {optimal_w, optimal_w};
        ExtendedPersistenceCalculator calculator;

        // Calculate divergent potential persistence
        Timer t5;
        auto divergent_result = calculator.computeExtendedPersistence(
            nhhd.D, mask_D, dx, nx_bins, ny_bins);
        #pragma omp atomic
        timing.persistence_div += t5.elapsed();

        // Calculate rotational potential persistence
        Timer t6;
        auto rotational_result = calculator.computeExtendedPersistence(
            -nhhd.Ru, mask_Ru, dx, nx_bins, ny_bins);
        #pragma omp atomic
        timing.persistence_rot += t6.elapsed();

        #pragma omp critical
        {
            std::cout << "Completed Mission " << mission << std::endl;
        }

        print_persistence_diagrams(mission, divergent_result);
        print_persistence_diagrams(mission, rotational_result);
    }

    // Print timing results
    std::cout << "\nExecution Times (seconds):" << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << "Data Loading:          " << timing.data_loading << std::endl;
    std::cout << "Vector Field:          " << timing.vector_field << std::endl;
    std::cout << "Divergence & Curl:     " << timing.div_curl << std::endl;
    std::cout << "Natural HHD:           " << timing.nhhd << std::endl;
    std::cout << "Persistence Divergent: " << timing.persistence_div << std::endl;
    std::cout << "Persistence Rotation:  " << timing.persistence_rot << std::endl;
    std::cout << "Total Time:            " 
              << timing.data_loading + timing.vector_field + timing.div_curl + 
                 timing.nhhd + timing.persistence_div + timing.persistence_rot 
              << std::endl;

    return 0;
}