#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <filesystem>
#include "data_loader.h"
#include "RGrid.h"
#include "VectorField.h"
#include "HHD.h"
#include "MaskedArray.h"
#include "GreensFunction.h"
#include "ExtendedPersistence.h"
#include "ChampVectorielGenerateur.h"

// Structure pour organiser les chemins de fichiers
struct OutputPaths {
    std::string base_dir;
    std::string velocity_dir;
    std::string density_dir;
    std::string weighted_dir;
    std::string persistence_dir;
    std::string persistence_weighted_dir;
    
    OutputPaths(const std::string& base = "output") : base_dir(base) {
        velocity_dir = base_dir + "/velocity";
        density_dir = base_dir + "/density";
        weighted_dir = base_dir + "/weighted";
        persistence_dir = base_dir + "/persistence/unweighted";
        persistence_weighted_dir = base_dir + "/persistence/weighted";
        
        // Création de la structure de répertoires
        for (const auto& dir : {
            base_dir,
            velocity_dir, velocity_dir + "/missions", velocity_dir + "/runs",
            density_dir, density_dir + "/missions", density_dir + "/runs",
            weighted_dir, weighted_dir + "/missions", weighted_dir + "/runs",
            base_dir + "/persistence",
            persistence_dir, persistence_dir + "/missions", persistence_dir + "/runs",
            persistence_weighted_dir, persistence_weighted_dir + "/missions", persistence_weighted_dir + "/runs"
        }) {
            std::filesystem::create_directory(dir);
        }
    }
};

void save_velocity_field(const std::string& mission_name, int run_index,
                        const MaskedArray<double>& velocity_x,
                        const MaskedArray<double>& velocity_y,
                        size_t nx_bins, size_t ny_bins,
                        const std::string& base_dir) {
    std::string subdir = (run_index == -1) ? "/missions/" : "/runs/";
    std::string filename = base_dir + subdir + "mission_" + mission_name;
    if (run_index >= 0) {
        filename += "_run_" + std::to_string(run_index);
    }
    filename += "_velocity.csv";
    
    std::ofstream file(filename);
    file << "x,y,vx,vy,mask\n";
    
    for(size_t i = 0; i < ny_bins; ++i) {
        for(size_t j = 0; j < nx_bins; ++j) {
            size_t idx = i * nx_bins + j;
            file << j << "," << i << ","
                 << velocity_x.data[idx] << "," 
                 << velocity_y.data[idx] << ","
                 << (!velocity_x.mask[idx]) << "\n";
        }
    }
}

void save_density_field(const std::string& mission_name, int run_index,
                       const MaskedArray<double>& density,
                       size_t nx_bins, size_t ny_bins,
                       const std::string& base_dir) {
    std::string subdir = (run_index == -1) ? "/missions/" : "/runs/";
    std::string filename = base_dir + subdir + "mission_" + mission_name;
    if (run_index >= 0) {
        filename += "_run_" + std::to_string(run_index);
    }
    filename += "_density.csv";
    
    std::ofstream file(filename);
    file << "x,y,density,mask\n";
    
    for(size_t i = 0; i < ny_bins; ++i) {
        for(size_t j = 0; j < nx_bins; ++j) {
            size_t idx = i * nx_bins + j;
            file << j << "," << i << ","
                 << density.data[idx] << ","
                 << (!density.mask[idx]) << "\n";
        }
    }
}

void save_persistence_to_csv(
    const ExtendedPersistenceCalculator::ExtendedPersistenceResult& div_result,
    const ExtendedPersistenceCalculator::ExtendedPersistenceResult& rot_result,
    const ExtendedPersistenceCalculator::ExtendedPersistenceResult& density_result,
    const std::string& filename) {
    
    std::ofstream file(filename);
    file << "type,diagram,field,birth,death\n";
    
    // Fonction helper pour écrire les résultats
    auto write_results = [&file](const auto& result, const std::string& field_type) {
        for(const auto& p : result.ord_h0)
            file << "ordinary,h0," << field_type << "," << p.first << "," << p.second << "\n";
        for(const auto& p : result.ord_h1)
            file << "ordinary,h1," << field_type << "," << p.first << "," << p.second << "\n";
        for(const auto& p : result.rel_h1)
            file << "relative,h1," << field_type << "," << p.first << "," << p.second << "\n";
        for(const auto& p : result.rel_h2)
            file << "relative,h2," << field_type << "," << p.first << "," << p.second << "\n";
        for(const auto& p : result.ext_plus_h0)
            file << "extended_plus,h0," << field_type << "," << p.first << "," << p.second << "\n";
        for(const auto& p : result.ext_minus_h1)
            file << "extended_minus,h1," << field_type << "," << p.first << "," << p.second << "\n";
    };
    
    write_results(div_result, "velocity_divergent");
    write_results(rot_result, "velocity_rotational");
    write_results(density_result, "density");
}

VectorField<double> create_vector_field(const DualFieldsData& fields, 
                                      size_t ny_bins, size_t nx_bins) {
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
    return vfield;
}

VectorField<double> create_weighted_vector_field(const DualFieldsData& fields, 
                                               size_t ny_bins, size_t nx_bins) {
    VectorField<double> vfield(std::vector<size_t>{ny_bins, nx_bins});
    for (size_t i = 0; i < ny_bins; ++i) {
        for (size_t j = 0; j < nx_bins; ++j) {
            size_t idx = i * nx_bins + j;
            // Multiplication de la vitesse par la densité
            vfield.u.data[idx] = fields.velocity.Ux(i, j) * fields.O(i, j);
            vfield.v.data[idx] = fields.velocity.Uy(i, j) * fields.O(i, j);
            vfield.u.mask[idx] = !fields.mask(i, j);
            vfield.v.mask[idx] = !fields.mask(i, j);
        }
    }
    return vfield;
}

MaskedArray<double> create_density_field(const DualFieldsData& fields,
                                       size_t ny_bins, size_t nx_bins) {
    MaskedArray<double> density_field(std::vector<size_t>{ny_bins, nx_bins});
    for (size_t i = 0; i < ny_bins; ++i) {
        for (size_t j = 0; j < nx_bins; ++j) {
            size_t idx = i * nx_bins + j;
            density_field.data[idx] = fields.O(i, j);
            density_field.mask[idx] = !fields.mask(i, j);
        }
    }
    return density_field;
}

struct PersistenceResults {
    ExtendedPersistenceCalculator::ExtendedPersistenceResult div_result;
    ExtendedPersistenceCalculator::ExtendedPersistenceResult rot_result;
    ExtendedPersistenceCalculator::ExtendedPersistenceResult density_result;
};

PersistenceResults compute_field_persistence(
    VectorField<double>& vfield,
    const MaskedArray<double>& density,
    const RGrid& rgrid,
    const std::array<double, 2>& dx,
    size_t nx_bins,
    size_t ny_bins) {
    
    // Calcul HHD
    vfield.need_divcurl(rgrid);
    std::vector<MaskedArray<double>> input_fields = {vfield.div, vfield.curl};
    naturalHHD<double> nhhd(input_fields, rgrid);

    // Préparation des masques
    MaskedArray<bool> mask_D(nhhd.D.shape);
    MaskedArray<bool> mask_Ru(nhhd.Ru.shape);
    MaskedArray<bool> density_mask(density.shape);
    
    mask_D.mask = nhhd.D.mask;
    mask_Ru.mask = nhhd.Ru.mask;
    density_mask.mask = density.mask;

    ExtendedPersistenceCalculator calculator;
    
    PersistenceResults results;
    results.div_result = calculator.computeExtendedPersistence(
        nhhd.D, mask_D, dx, nx_bins, ny_bins);
    results.rot_result = calculator.computeExtendedPersistence(
        -nhhd.Ru, mask_Ru, dx, nx_bins, ny_bins);
    results.density_result = calculator.computeExtendedPersistence(
        density, density_mask, dx, nx_bins, ny_bins);
        
    return results;
}


void process_fields(const DualFieldsData& fields, const std::string& mission_name,
                   int run_index, const RGrid& rgrid, const std::array<double, 2>& dx,
                   size_t nx_bins, size_t ny_bins, const OutputPaths& paths) {
    
    // Création des champs
    auto vfield = create_vector_field(fields, ny_bins, nx_bins);
    auto weighted_vfield = create_weighted_vector_field(fields, ny_bins, nx_bins);
    auto density = create_density_field(fields, ny_bins, nx_bins);

    // Sauvegarde des champs séparés
    save_velocity_field(mission_name, run_index, vfield.u, vfield.v, 
                       nx_bins, ny_bins, paths.velocity_dir);
    save_density_field(mission_name, run_index, density, 
                      nx_bins, ny_bins, paths.density_dir);
    
    // Sauvegarde des champs pondérés
    std::string weighted_name = mission_name + "_weighted";
    save_velocity_field(weighted_name, run_index, weighted_vfield.u, weighted_vfield.v,
                       nx_bins, ny_bins, paths.weighted_dir);

    // Calcul de la persistance pour les deux types de champs
    auto std_persistence = compute_field_persistence(
        vfield, density, rgrid, dx, nx_bins, ny_bins);
    auto weighted_persistence = compute_field_persistence(
        weighted_vfield, density, rgrid, dx, nx_bins, ny_bins);

    // Construction des chemins pour la persistance
    std::string subdir = (run_index == -1) ? "/missions/" : "/runs/";
    
    // Chemin pour la persistance non pondérée
    std::string unweighted_filename = paths.persistence_dir + subdir + mission_name;
    if (run_index >= 0) {
        unweighted_filename += "_run_" + std::to_string(run_index);
    }
    unweighted_filename += "_persistence.csv";
    
    // Chemin pour la persistance pondérée
    std::string weighted_filename = paths.persistence_weighted_dir + subdir + mission_name;
    if (run_index >= 0) {
        weighted_filename += "_run_" + std::to_string(run_index);
    }
    weighted_filename += "_persistence.csv";

    // Sauvegarde des résultats de persistance séparément
    save_persistence_to_csv(
        std_persistence.div_result,
        std_persistence.rot_result,
        std_persistence.density_result,
        unweighted_filename
    );
    
    save_persistence_to_csv(
        weighted_persistence.div_result,
        weighted_persistence.rot_result,
        weighted_persistence.density_result,
        weighted_filename
    );
}

int main() {
    const double optimal_w = 0.2043;
    const double R_ARENA_SIZE = 2 * 1.231;

    std::cout << "Starting data processing..." << std::endl;

    // Charge les données
    DataLoader data_loader(".");
    auto [all_mission_data, missions] = data_loader.loadAllMissions();

    if (all_mission_data.empty()) {
        std::cerr << "No mission data loaded." << std::endl;
        return 1;
    }

    // Configuration grille
    size_t nx_bins = static_cast<size_t>(2 * std::ceil(0.5 * R_ARENA_SIZE / optimal_w));
    size_t ny_bins = nx_bins;
    RGrid rgrid(nx_bins, ny_bins, optimal_w, optimal_w);
    std::array<double, 2> dx = {optimal_w, optimal_w};

    // Initialisation des chemins de sortie
    OutputPaths paths;

    // Traite chaque mission
    for (size_t mission_idx = 0; mission_idx < all_mission_data.size(); ++mission_idx) {
        const auto& mission_data = all_mission_data[mission_idx];
        std::string mission_name = missions[mission_idx];
        std::cout << "\nProcessing Mission " << mission_name << std::endl;

        // Traitement de la mission complète (moyenne des runs)
        auto mission_fields = ChampVectorielGenerateur::generate_mission_fields(
            mission_data, optimal_w);
        process_fields(mission_fields, mission_name, -1, rgrid, dx, nx_bins, ny_bins, paths);

        // Traitement de chaque run individuellement
        for(size_t run_idx = 0; run_idx < mission_data.size(); ++run_idx) {
            std::cout << "Processing Run " << run_idx << std::endl;
            
            auto run_fields = ChampVectorielGenerateur::generate_single_run_fields(
                mission_data[run_idx], optimal_w);
            
            process_fields(run_fields, mission_name, run_idx, rgrid, dx, nx_bins, ny_bins, paths);
        }
    }

    std::cout << "Processing complete!" << std::endl;
    return 0;
}
