#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

DataLoader::DataLoader(const std::string& base_directory) 
    : base_directory_(base_directory) {}

std::pair<std::vector<std::vector<RunData>>, std::vector<std::string>> DataLoader::loadAllMissions() {
    std::string cache_file = getBinaryFilePath();
    
    // Try to load from cache first
    if (fileExists(cache_file)) {
        std::cout << "Found cached data, loading from file..." << std::endl;
        auto cached_data = loadFromFile(cache_file);
        if (!cached_data.empty()) {
            std::cout << "Successfully loaded cached data" << std::endl;
            return {cached_data, {}};
        }
    }

    std::cout << "No valid cache found, loading data from source..." << std::endl;
    
    std::vector<std::vector<RunData>> all_mission_data;
    std::vector<std::string> missions, best_fsms;

    if (!readBestFSMs(missions, best_fsms)) {
        std::cerr << "Failed to read best FSMs" << std::endl;
        return {all_mission_data, missions};
    }

    // Process each mission
    for (size_t i = 0; i < missions.size(); ++i) {
        const std::string& mission = missions[i];
        const std::string& best_fsm = best_fsms[i];
        std::string mission_name = mission.substr(0, mission.find('_'));
        std::string autocontrollers_path = base_directory_ + "/Design/" + mission_name + 
                                         "/Chocolate/autocontrollers.txt";

        int fsm_number = findFSMNumber(best_fsm, autocontrollers_path);
        if (fsm_number != -1) {
            std::vector<RunData> mission_data;
            if (readPerformanceData(mission_name, fsm_number, mission_data)) {
                all_mission_data.push_back(mission_data);
                std::cout << "Successfully loaded data for " << mission_name << std::endl;
            } else {
                std::cerr << "No valid data found for " << mission_name << std::endl;
            }
        } else {
            std::cerr << "Could not find FSM number for " << mission_name << std::endl;
        }
    }

    // Save to cache if data was loaded successfully
    if (!all_mission_data.empty()) {
        std::cout << "Saving data to cache..." << std::endl;
        if (saveToFile(all_mission_data, cache_file)) {
            std::cout << "Successfully saved data to cache" << std::endl;
        }
    }

    return {all_mission_data, missions};
}

bool DataLoader::saveToFile(const std::vector<std::vector<RunData>>& data, 
                          const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    // Write number of missions
    size_t num_missions = data.size();
    file.write(reinterpret_cast<const char*>(&num_missions), sizeof(num_missions));

    for (const auto& mission : data) {
        // Write number of runs
        size_t num_runs = mission.size();
        file.write(reinterpret_cast<const char*>(&num_runs), sizeof(num_runs));

        for (const auto& run : mission) {
            // Write run metadata
            file.write(reinterpret_cast<const char*>(&run.objective_function), sizeof(double));
            file.write(reinterpret_cast<const char*>(&run.num_robots), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&run.num_time_steps), sizeof(size_t));

            // Write array sizes
            size_t x_size = run.x.size();
            size_t y_size = run.y.size();
            size_t vx_size = run.vx.size();
            size_t vy_size = run.vy.size();
            
            file.write(reinterpret_cast<const char*>(&x_size), sizeof(size_t));
            
            // Write data arrays
            for (const auto& row : run.x) {
                size_t row_size = row.size();
                file.write(reinterpret_cast<const char*>(&row_size), sizeof(size_t));
                file.write(reinterpret_cast<const char*>(row.data()), row_size * sizeof(double));
            }
            for (const auto& row : run.y) {
                size_t row_size = row.size();
                file.write(reinterpret_cast<const char*>(&row_size), sizeof(size_t));
                file.write(reinterpret_cast<const char*>(row.data()), row_size * sizeof(double));
            }
            for (const auto& row : run.vx) {
                size_t row_size = row.size();
                file.write(reinterpret_cast<const char*>(&row_size), sizeof(size_t));
                file.write(reinterpret_cast<const char*>(row.data()), row_size * sizeof(double));
            }
            for (const auto& row : run.vy) {
                size_t row_size = row.size();
                file.write(reinterpret_cast<const char*>(&row_size), sizeof(size_t));
                file.write(reinterpret_cast<const char*>(row.data()), row_size * sizeof(double));
            }
        }
    }

    return true;
}

std::vector<std::vector<RunData>> DataLoader::loadFromFile(const std::string& filename) const {
    std::vector<std::vector<RunData>> data;
    std::ifstream file(filename, std::ios::binary);
    if (!file) return data;

    try {
        size_t num_missions;
        file.read(reinterpret_cast<char*>(&num_missions), sizeof(num_missions));
        data.resize(num_missions);

        for (auto& mission : data) {
            size_t num_runs;
            file.read(reinterpret_cast<char*>(&num_runs), sizeof(num_runs));
            mission.resize(num_runs);

            for (auto& run : mission) {
                // Read run metadata
                file.read(reinterpret_cast<char*>(&run.objective_function), sizeof(double));
                file.read(reinterpret_cast<char*>(&run.num_robots), sizeof(size_t));
                file.read(reinterpret_cast<char*>(&run.num_time_steps), sizeof(size_t));

                // Read array sizes
                size_t x_size;
                file.read(reinterpret_cast<char*>(&x_size), sizeof(size_t));
                
                // Read data arrays
                run.x.resize(x_size);
                run.y.resize(x_size);
                run.vx.resize(x_size);
                run.vy.resize(x_size);

                for (auto& row : run.x) {
                    size_t row_size;
                    file.read(reinterpret_cast<char*>(&row_size), sizeof(size_t));
                    row.resize(row_size);
                    file.read(reinterpret_cast<char*>(row.data()), row_size * sizeof(double));
                }
                for (auto& row : run.y) {
                    size_t row_size;
                    file.read(reinterpret_cast<char*>(&row_size), sizeof(size_t));
                    row.resize(row_size);
                    file.read(reinterpret_cast<char*>(row.data()), row_size * sizeof(double));
                }
                for (auto& row : run.vx) {
                    size_t row_size;
                    file.read(reinterpret_cast<char*>(&row_size), sizeof(size_t));
                    row.resize(row_size);
                    file.read(reinterpret_cast<char*>(row.data()), row_size * sizeof(double));
                }
                for (auto& row : run.vy) {
                    size_t row_size;
                    file.read(reinterpret_cast<char*>(&row_size), sizeof(size_t));
                    row.resize(row_size);
                    file.read(reinterpret_cast<char*>(row.data()), row_size * sizeof(double));
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading cache file: " << e.what() << std::endl;
        data.clear();
    }

    return data;
}

bool DataLoader::readBestFSMs(std::vector<std::string>& missions, 
                            std::vector<std::string>& best_fsms) const {
    std::string file_path = base_directory_ + "/best_fsms.csv";
    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        std::cerr << "Error: File " << file_path << " does not exist." << std::endl;
        return false;
    }
    
    std::string line;
    // Skip header
    std::getline(infile, line);
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string mission, fsm;
        std::getline(ss, mission, ',');
        std::getline(ss, fsm);
        missions.push_back(mission);
        best_fsms.push_back(fsm);
    }
    return !missions.empty();
}

int DataLoader::findFSMNumber(const std::string& best_fsm, 
                            const std::string& autocontrollers_path) const {
    std::ifstream infile(autocontrollers_path);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << autocontrollers_path << std::endl;
        return -1;
    }
    
    std::string line;
    int line_number = 0;
    while (std::getline(infile, line)) {
        if (line.find(best_fsm) != std::string::npos) {
            return line_number;
        }
        line_number++;
    }
    return -1;
}

bool DataLoader::readPerformanceData(const std::string& mission_name, 
                                   int fsm_number, 
                                   std::vector<RunData>& all_data) const {
    std::string base_path = base_directory_ + "/argoslogs_chocosim/" + mission_name + 
                           "/Chocolate/simulation/" + std::to_string(fsm_number);
    
    if (!fs::exists(base_path)) {
        std::cerr << "Warning: Path does not exist: " << base_path << std::endl;
        return false;
    }

    for (const auto& entry : fs::directory_iterator(base_path)) {
        if (entry.path().extension() == ".csv") {
            std::ifstream infile(entry.path());
            if (!infile.is_open()) {
                std::cerr << "Warning: Cannot open file " << entry.path() << std::endl;
                continue;
            }

            std::string line;
            std::vector<std::string> lines;
            while (std::getline(infile, line)) {
                lines.push_back(line);
            }

            // Parse positions
            size_t positions_end = lines.size();
            for (size_t i = 1; i < lines.size(); ++i) {
                if (lines[i].find("# SDBC") != std::string::npos) {
                    positions_end = i;
                    break;
                }
            }

            std::vector<std::vector<double>> positions;
            for (size_t i = 1; i < positions_end; ++i) {
                std::stringstream ss(lines[i]);
                std::string token;
                std::vector<double> pos;
                while (std::getline(ss, token, ',')) {
                    try {
                        pos.push_back(std::stod(token));
                    } catch (const std::exception& e) {
                        continue;
                    }
                }
                if (!pos.empty()) {
                    positions.push_back(pos);
                }
            }

            if (positions.empty()) {
                std::cerr << "Warning: No valid positions in file " << entry.path() << std::endl;
                continue;
            }

            RunData run_data;
            run_data.num_robots = positions.size();
            run_data.num_time_steps = positions[0].size() / 2;

            // Extract objective function
            try {
                run_data.objective_function = std::stod(lines.back());
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid objective function in file " << entry.path() << std::endl;
                continue;
            }
            
            // Initialize arrays with correct dimensions
            run_data.x.resize(run_data.num_time_steps);
            run_data.y.resize(run_data.num_time_steps);
            run_data.vx.resize(run_data.num_time_steps - 1);
            run_data.vy.resize(run_data.num_time_steps - 1);

            for (auto& row : run_data.x) row.resize(run_data.num_robots);
            for (auto& row : run_data.y) row.resize(run_data.num_robots);
            for (auto& row : run_data.vx) row.resize(run_data.num_robots);
            for (auto& row : run_data.vy) row.resize(run_data.num_robots);

            // Fill positions
            for (size_t t = 0; t < run_data.num_time_steps; ++t) {
                for (size_t i = 0; i < run_data.num_robots; ++i) {
                    run_data.x[t][i] = positions[i][2 * t];
                    run_data.y[t][i] = positions[i][2 * t + 1];
                }
            }

            // Compute velocities
            const double R_DT = 0.1; // Time step
            for (size_t t = 0; t < run_data.num_time_steps - 1; ++t) {
                for (size_t i = 0; i < run_data.num_robots; ++i) {
                    run_data.vx[t][i] = (run_data.x[t + 1][i] - run_data.x[t][i]) / R_DT;
                    run_data.vy[t][i] = (run_data.y[t + 1][i] - run_data.y[t][i]) / R_DT;
                }
            }

            all_data.push_back(run_data);
        }
    }

    return !all_data.empty();
}

std::string DataLoader::getBinaryFilePath() const {
    return base_directory_ + "/mission_data_cache.bin";
}

bool DataLoader::fileExists(const std::string& filename) const {
    return fs::exists(filename);
}