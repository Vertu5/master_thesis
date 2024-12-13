#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <limits>
#include "VectorField.h"
#include "ChampVectorielGenerateur.h"
#include "HHD.h"  
#include "ExtendedPersistence.h"
#include "ChampVectorielData.h"
#include "hera/wasserstein.h"

double distance_to_diagonal(const std::pair<double, double>& point) {
    return std::abs(point.second - point.first);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <simulation.csv> <expert_behavior_file>" << std::endl;
        return 1;
    }

    const double OPTIMAL_W = 0.2043;

    // Lire les données de simulation
    std::ifstream simulation_file(argv[1]);
    if (!simulation_file.is_open()) {
        std::cerr << "Échec de l'ouverture du fichier de simulation : " << argv[1] << std::endl;
        return 1;
    }

    // Parser les données de simulation
    std::string line;
    bool reading_positions = false;
    std::vector<std::vector<double>> positions;

    while (std::getline(simulation_file, line)) {
        if (line == "# Positions") {
            reading_positions = true;
            continue;
        }
        if (reading_positions && line[0] == '#') {
            break;
        }
        if (reading_positions) {
            std::vector<double> row_data;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ',')) {
                try {
                    row_data.push_back(std::stod(value));
                } catch (...) {
                    std::cerr << "Format de nombre invalide dans la ligne : " << line << std::endl;
                    return 1;
                }
            }
            if (!row_data.empty()) {
                positions.push_back(row_data);
            }
        }
    }

    if (positions.empty()) {
        std::cerr << "Aucune donnée de position valide trouvée" << std::endl;
        return 1;
    }

    // Préparer les données
    size_t num_robots = positions.size();
    size_t num_timesteps = positions[0].size() / 2;

    RunData run_data;
    run_data.num_robots = num_robots;
    run_data.num_time_steps = num_timesteps;

    run_data.x.resize(num_timesteps);
    run_data.y.resize(num_timesteps);
    run_data.vx.resize(num_timesteps - 1);
    run_data.vy.resize(num_timesteps - 1);

    for(auto& row : run_data.x) row.resize(num_robots);
    for(auto& row : run_data.y) row.resize(num_robots);
    for(auto& row : run_data.vx) row.resize(num_robots);
    for(auto& row : run_data.vy) row.resize(num_robots);

    // Remplir positions et calculer vitesses
    const double R_DT = 0.1;
    for(size_t t = 0; t < num_timesteps; ++t) {
        for(size_t r = 0; r < num_robots; ++r) {
            run_data.x[t][r] = positions[r][2*t];
            run_data.y[t][r] = positions[r][2*t + 1];
            
            if(t < num_timesteps - 1) {
                // Calcul des vitesses
                double dx = (positions[r][2*(t+1)] - positions[r][2*t]) / R_DT;
                double dy = (positions[r][2*(t+1)+1] - positions[r][2*t+1]) / R_DT;
                
                // Normalisation
                double magnitude = std::sqrt(dx*dx + dy*dy);
                if (magnitude < 1e-10) magnitude = 1e-10;
                
                run_data.vx[t][r] = dx / magnitude;
                run_data.vy[t][r] = dy / magnitude;
            }
        }
    }

    // Créer un vecteur d'une seule mission
    std::vector<RunData> single_mission = {run_data};
    
    // Générer les champs O et Uv
    auto fields = ChampVectorielGenerateur::generate_mission_fields(single_mission, OPTIMAL_W);

    // Configurer la grille
    size_t nx_bins = static_cast<size_t>(2 * std::ceil(0.5 * ChampVectorielGenerateur::ARENA_SIZE / OPTIMAL_W));
    size_t ny_bins = nx_bins;
    RGrid rgrid(nx_bins, ny_bins, OPTIMAL_W, OPTIMAL_W);

    // Créer VectorField pour la vitesse
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

    // Calcul pour le champ de vitesse
    vfield.need_divcurl(rgrid);
    std::vector<MaskedArray<double>> input_fields = {vfield.div, vfield.curl};
    naturalHHD<double> nhhd(input_fields, rgrid);

    // Préparation pour la persistance
    std::array<double, 2> persistence_dx = {OPTIMAL_W, OPTIMAL_W};
    
    // Masques pour HHD
    MaskedArray<bool> mask_D(nhhd.D.shape);
    mask_D.mask = nhhd.D.mask;
    MaskedArray<bool> mask_Ru(nhhd.Ru.shape);
    mask_Ru.mask = nhhd.Ru.mask;

    // Masque pour densité
    MaskedArray<double> density_field(std::vector<size_t>{ny_bins, nx_bins});
    MaskedArray<bool> density_mask(std::vector<size_t>{ny_bins, nx_bins});
    
    // Copier le champ de densité
    for(size_t i = 0; i < ny_bins; ++i) {
        for(size_t j = 0; j < nx_bins; ++j) {
            size_t idx = i * nx_bins + j;
            density_field.data[idx] = fields.O(i, j);
            density_mask.mask[idx] = !fields.mask(i, j);
        }
    }

    // Calculer la persistance
    ExtendedPersistenceCalculator calculator;
    
    // Pour le champ de vitesse
    auto divergent_result = calculator.computeExtendedPersistence(
        nhhd.D, mask_D, persistence_dx, nx_bins, ny_bins);
    auto rotational_result = calculator.computeExtendedPersistence(
        -nhhd.Ru, mask_Ru, persistence_dx, nx_bins, ny_bins);
        
    // Pour le champ de densité
    // auto density_result = calculator.computeExtendedPersistence(
    //     density_field, density_mask, persistence_dx, nx_bins, ny_bins);

    // Collecter tous les diagrammes (18)
    std::vector<PersistenceDiagram> imitation_diagrams = {
        // Champ de vitesse (12)
        divergent_result.ord_h0, divergent_result.ord_h1,
        divergent_result.rel_h1, divergent_result.rel_h2,
        divergent_result.ext_plus_h0, divergent_result.ext_minus_h1,
        // rotational_result.ord_h0, rotational_result.ord_h1,
        // rotational_result.rel_h1, rotational_result.rel_h2,
        // rotational_result.ext_plus_h0, rotational_result.ext_minus_h1,
        //Champ de densité (6)
        // density_result.ord_h0, density_result.ord_h1,
        // density_result.rel_h1, density_result.rel_h2,
        // density_result.ext_plus_h0, density_result.ext_minus_h1
    };

    // Filtrer les paires (0,0)
    for(auto& diagram : imitation_diagrams) {
        diagram.erase(
            std::remove_if(diagram.begin(), diagram.end(),
                [](const std::pair<double, double>& p) { 
                    return p.first == 0.0 && p.second == 0.0; 
                }),
            diagram.end());
    }

    // Lire les diagrammes experts
    std::ifstream expert_file(argv[2], std::ios::binary);
    if (!expert_file.is_open()) {
        std::cerr << "Échec de l'ouverture du fichier expert : " << argv[2] << std::endl;
        return 1;
    }

    // not 12 but len imitation_diagrams
    
    std::vector<PersistenceDiagram> expert_diagrams(imitation_diagrams.size());
    for (size_t i = 0; i < expert_diagrams.size(); ++i) {
        size_t num_pairs;
        expert_file.read(reinterpret_cast<char*>(&num_pairs), sizeof(size_t));
        if (expert_file.fail()) {
            std::cerr << "Erreur lecture diagramme " << i+1 << std::endl;
            return 1;
        }

        expert_diagrams[i].reserve(num_pairs);
        for (size_t j = 0; j < num_pairs; ++j) {
            double birth, death;
            expert_file.read(reinterpret_cast<char*>(&birth), sizeof(double));
            expert_file.read(reinterpret_cast<char*>(&death), sizeof(double));
            
            if (expert_file.fail()) {
                std::cerr << "Erreur lecture paire " << j+1 << " diagramme " << i+1 << std::endl;
                return 1;
            }

            if(!(birth == 0.0 && death == 0.0)) {
                expert_diagrams[i].emplace_back(birth, death);
            }
        }
    }
    expert_file.close();

    // Calcul des distances de Wasserstein
    double total_distance = 0.0;
    hera::AuctionParams<double> wasserstein_params;
    wasserstein_params.wasserstein_power = 9.0;
    wasserstein_params.delta = 0.01;

    for (size_t i = 0; i < imitation_diagrams.size(); ++i) {
        if (expert_diagrams[i].empty() && imitation_diagrams[i].empty()) {
            continue;
        }
        
        if (expert_diagrams[i].empty() || imitation_diagrams[i].empty()) {
            double distance = 0.0;
            const auto& non_empty_diagram = expert_diagrams[i].empty() ? 
                imitation_diagrams[i] : expert_diagrams[i];
                
            for (const auto& pair : non_empty_diagram) {
                distance += distance_to_diagonal(pair);
            }
            total_distance += distance;
            continue;
        }

        try { 
            double distance = hera::wasserstein_dist(
                expert_diagrams[i], imitation_diagrams[i], wasserstein_params);
            total_distance += distance;
        }
        catch (const std::exception& e) {
            total_distance += std::numeric_limits<double>::infinity();
        }
    }

    std::cout << total_distance << std::endl;
    return 0;
}
