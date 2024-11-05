#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

bool read_best_fsms(const std::string& file_path, std::vector<std::string>& missions, std::vector<std::string>& best_fsms) {
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
    return true;
}

int find_fsm_number(const std::string& fsm, const std::string& autocontrollers_path) {
    std::ifstream infile(autocontrollers_path);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << autocontrollers_path << std::endl;
        return -1;
    }
    std::string line;
    int line_number = 0;
    while (std::getline(infile, line)) {
        if (line.find(fsm) != std::string::npos) {
            return line_number;
        }
        line_number++;
    }
    return -1;
}

bool read_performance_data(const std::string& mission_name, int fsm_number, std::vector<RunData>& all_data) {
    std::string base_path = "argoslogs_chocosim/" + mission_name + "/Chocolate/simulation/" + std::to_string(fsm_number);
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

            // Extract objective function
            double objective_function = 0.0;
            try {
                objective_function = std::stod(lines.back());
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid objective function in file " << entry.path() << std::endl;
                continue;
            }

            if (positions.empty()) {
                std::cerr << "Warning: No valid positions in file " << entry.path() << std::endl;
                continue;
            }

            size_t num_robots = positions.size();
            size_t num_time_steps = positions[0].size() / 2;

            RunData run_data;
            run_data.x.resize(num_time_steps, std::vector<double>(num_robots));
            run_data.y.resize(num_time_steps, std::vector<double>(num_robots));
            run_data.vx.resize(num_time_steps - 1, std::vector<double>(num_robots));
            run_data.vy.resize(num_time_steps - 1, std::vector<double>(num_robots));
            run_data.objective_function = objective_function;
            run_data.num_robots = num_robots;
            run_data.num_time_steps = num_time_steps;

            // Fill positions
            for (size_t i = 0; i < num_robots; ++i) {
                for (size_t t = 0; t < num_time_steps; ++t) {
                    run_data.x[t][i] = positions[i][2 * t];
                    run_data.y[t][i] = positions[i][2 * t + 1];
                }
            }

            // Compute velocities
            double R_DT = 0.1; // Time step
            for (size_t t = 0; t < num_time_steps - 1; ++t) {
                for (size_t i = 0; i < num_robots; ++i) {
                    run_data.vx[t][i] = (run_data.x[t + 1][i] - run_data.x[t][i]) / R_DT;
                    run_data.vy[t][i] = (run_data.y[t + 1][i] - run_data.y[t][i]) / R_DT;
                }
            }

            all_data.push_back(run_data);
        }
    }

    if (all_data.empty()) {
        std::cerr << "Warning: No valid data found for mission " << mission_name << ", FSM number " << fsm_number << std::endl;
        return false;
    }

    return true;
}
