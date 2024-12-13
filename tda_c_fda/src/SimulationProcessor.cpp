#include "SimulationProcessor.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include "ChampVectorielGenerateur.h"

SimulationProcessor::SimulationProcessor(const std::string& simulation_dir, 
                                       const std::string& output_dir)
    : simulation_dir_(simulation_dir), 
      output_paths_(output_dir) {}

std::vector<std::string> SimulationProcessor::findAvailableMissions() const {
    std::vector<std::string> missions;
    
    for(const auto& entry : fs::directory_iterator(simulation_dir_)) {
        if(entry.is_directory() && entry.path().filename() != "log_seeds.txt") {
            missions.push_back(entry.path().filename().string());
        }
    }
    
    return missions;
}

RunData SimulationProcessor::readSimulationCSV(const fs::path& file_path) {
    RunData run_data;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path.string());
    }

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    // Trouver la fin des positions
    size_t positions_end = lines.size();
    for (size_t i = 1; i < lines.size(); ++i) {
        if (lines[i].find("# SDBC") != std::string::npos) {
            positions_end = i;
            break;
        }
    }

    // Parser les positions
    std::vector<std::vector<double>> positions;
    for (size_t i = 1; i < positions_end; ++i) {
        std::stringstream ss(lines[i]);
        std::string token;
        std::vector<double> pos;
        while (std::getline(ss, token, ',')) {
            try {
                pos.push_back(std::stod(token));
            } catch (...) {
                continue;
            }
        }
        if (!pos.empty()) {
            positions.push_back(pos);
        }
    }

    if (positions.empty()) {
        throw std::runtime_error("No valid positions found in file: " + file_path.string());
    }

    // Configurer les dimensions
    run_data.num_robots = positions.size();
    run_data.num_time_steps = positions[0].size() / 2;

    // Initialiser les tableaux
    run_data.x.resize(run_data.num_time_steps);
    run_data.y.resize(run_data.num_time_steps);
    run_data.vx.resize(run_data.num_time_steps - 1);
    run_data.vy.resize(run_data.num_time_steps - 1);

    for (auto& row : run_data.x) row.resize(run_data.num_robots);
    for (auto& row : run_data.y) row.resize(run_data.num_robots);
    for (auto& row : run_data.vx) row.resize(run_data.num_robots);
    for (auto& row : run_data.vy) row.resize(run_data.num_robots);

    // Remplir les positions
    for (size_t t = 0; t < run_data.num_time_steps; ++t) {
        for (size_t i = 0; i < run_data.num_robots; ++i) {
            run_data.x[t][i] = positions[i][2 * t];
            run_data.y[t][i] = positions[i][2 * t + 1];
        }
    }

    // Calculer les vitesses
    const double TIME_STEP = 0.1;
    for (size_t t = 0; t < run_data.num_time_steps - 1; ++t) {
        for (size_t i = 0; i < run_data.num_robots; ++i) {
            run_data.vx[t][i] = (run_data.x[t + 1][i] - run_data.x[t][i]) / TIME_STEP;
            run_data.vy[t][i] = (run_data.y[t + 1][i] - run_data.y[t][i]) / TIME_STEP;
        }
    }

    return run_data;
}

std::vector<RunData> SimulationProcessor::readPFSMDirectory(const fs::path& pfsm_path) {
    std::vector<RunData> runs;
    
    if (!fs::exists(pfsm_path)) {
        std::cerr << "PFSM directory does not exist: " << pfsm_path << std::endl;
        return runs;
    }

    for(const auto& entry : fs::directory_iterator(pfsm_path)) {
        if(entry.path().extension() == ".csv") {
            try {
                runs.push_back(readSimulationCSV(entry.path()));
            }
            catch(const std::exception& e) {
                std::cerr << "Error reading " << entry.path() << ": " << e.what() << std::endl;
            }
        }
    }
    
    return runs;
}

void SimulationProcessor::computeAndSavePersistence(
    const RunData& data,
    const std::string& mission_name,
    const std::string& type,
    int pfsm_index,
    int run_index)
{
    // Configuration de la grille
    size_t nx_bins = static_cast<size_t>(2 * std::ceil(0.5 * arena_size / optimal_w));
    size_t ny_bins = nx_bins;
    RGrid rgrid(nx_bins, ny_bins, optimal_w, optimal_w);
    std::array<double, 2> dx = {optimal_w, optimal_w};

    // Génération des champs
    auto fields = ChampVectorielGenerateur::generate_single_run_fields(data, optimal_w);

    // Création des champs vectoriels
    VectorField<double> vfield(std::vector<size_t>{ny_bins, nx_bins});
    VectorField<double> weighted_vfield(std::vector<size_t>{ny_bins, nx_bins});

    // Remplissage des champs
    for (size_t i = 0; i < ny_bins; ++i) {
        for (size_t j = 0; j < nx_bins; ++j) {
            size_t idx = i * nx_bins + j;
            // Champ standard
            vfield.u.data[idx] = fields.velocity.Ux(i, j);
            vfield.v.data[idx] = fields.velocity.Uy(i, j);
            vfield.u.mask[idx] = !fields.mask(i, j);
            vfield.v.mask[idx] = !fields.mask(i, j);
            
            // Champ pondéré
            weighted_vfield.u.data[idx] = fields.velocity.Ux(i, j) * fields.O(i, j);
            weighted_vfield.v.data[idx] = fields.velocity.Uy(i, j) * fields.O(i, j);
            weighted_vfield.u.mask[idx] = !fields.mask(i, j);
            weighted_vfield.v.mask[idx] = !fields.mask(i, j);
        }
    }

    // Construire les chemins de fichiers
    fs::path unweighted_path, weighted_path;
    if (type == "Expert") {
        unweighted_path = output_paths_.unweighted_missions / (mission_name + "_expert.csv");
        weighted_path = output_paths_.weighted_missions / (mission_name + "_expert.csv");
    } else {
        std::string filename = mission_name + "_pfsm" + std::to_string(pfsm_index) + 
                             "_run" + std::to_string(run_index) + ".csv";
        unweighted_path = output_paths_.unweighted_runs / filename;
        weighted_path = output_paths_.weighted_runs / filename;
    }

    // Traitement des deux types de champs
    struct FieldInfo {
        VectorField<double>* field;
        fs::path path;
    };

    std::vector<FieldInfo> field_configs = {
        {&vfield, unweighted_path},
        {&weighted_vfield, weighted_path}
    };

    for (const auto& config : field_configs) {
        // Calcul HHD
        config.field->need_divcurl(rgrid);
        std::vector<MaskedArray<double>> input_fields;
        input_fields.push_back(config.field->div);
        input_fields.push_back(config.field->curl);
        naturalHHD<double> nhhd(input_fields, rgrid);

        // Préparation des masques
        MaskedArray<bool> mask_D(nhhd.D.shape);
        mask_D.mask = nhhd.D.mask;
        MaskedArray<bool> mask_Ru(nhhd.Ru.shape);
        mask_Ru.mask = nhhd.Ru.mask;

        // Calcul de l'extended persistence
        ExtendedPersistenceCalculator calculator;
        auto divergent_result = calculator.computeExtendedPersistence(
            nhhd.D, mask_D, dx, nx_bins, ny_bins);
        auto rotational_result = calculator.computeExtendedPersistence(
            -nhhd.Ru, mask_Ru, dx, nx_bins, ny_bins);

        // Sauvegarde des résultats
        std::ofstream file(config.path);
        file << "type,diagram,field,birth,death\n";
        
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
        
        write_results(divergent_result, "velocity_divergent");
        write_results(rotational_result, "velocity_rotational");
    }
}

void SimulationProcessor::processMission(const std::string& mission_name) {
    std::cout << "Processing mission: " << mission_name << std::endl;

    // Traiter l'Expert (toujours dans le dossier 0)
    fs::path expert_path = simulation_dir_ / mission_name / "Expert" / "simulation" / "0";
    if (fs::exists(expert_path)) {
        std::cout << "  Processing Expert data..." << std::endl;
        auto expert_runs = readPFSMDirectory(expert_path);
        if (!expert_runs.empty()) {
            computeAndSavePersistence(expert_runs[0], mission_name, "Expert");
        } else {
            std::cerr << "No expert data found for mission " << mission_name << std::endl;
        }
    }

    // Traiter les Imitations (dossiers 0-7)
    fs::path imitation_base = simulation_dir_ / mission_name / "Imitation" / "simulation";
    if (fs::exists(imitation_base)) {
        for (int pfsm = 0; pfsm < 8; ++pfsm) {
            fs::path pfsm_path = imitation_base / std::to_string(pfsm);
            if (!fs::exists(pfsm_path)) continue;

            std::cout << "  Processing PFSM " << pfsm << "..." << std::endl;
            auto pfsm_runs = readPFSMDirectory(pfsm_path);
            
            for (size_t run_idx = 0; run_idx < pfsm_runs.size(); ++run_idx) {
                computeAndSavePersistence(
                    pfsm_runs[run_idx], 
                    mission_name, 
                    "Imitation",
                    pfsm,
                    run_idx
                );
            }
            
            if (pfsm_runs.empty()) {
                std::cerr << "No runs found for PFSM " << pfsm << std::endl;
            } else {
                std::cout << "    Processed " << pfsm_runs.size() << " runs" << std::endl;
            }
        }
    }
}

void SimulationProcessor::processAllMissions() {
    auto missions = findAvailableMissions();
    
    if (missions.empty()) {
        std::cerr << "No missions found in directory: " << simulation_dir_ << std::endl;
        return;
    }

    std::cout << "Found " << missions.size() << " missions to process:" << std::endl;
    for (const auto& mission : missions) {
        std::cout << "- " << mission << std::endl;
    }
    std::cout << std::endl;

    for (const auto& mission : missions) {
        try {
            processMission(mission);
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing mission " << mission << ": " << e.what() << std::endl;
        }
    }
}
