#include "ArgosSimulation_forcomparation.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <random>
#include <map>
#include <algorithm>
#include <thread>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::vector<std::string> missions = {
        "Homing",
        "homingDensityVelocity", 
        "homingVelocityovertime", 
        "homingwithjustDivergence"
    };
    
    std::map<std::string, std::string> mission_lowercase_map = {
        {"Homing", "homing"},
        {"homingDensityVelocity", "homingDensityVelocity"},
        {"homingVelocityovertime", "homingVelocityovertime"},
        {"homingwithjustDivergence", "homingwithjustDivergence"}
    };
    
    std::vector<std::string> methods = {"Chocolate"};
    std::vector<std::string> environments = {"simulation"};

    // Create directory for logs
    fs::remove_all("HomingSimulationCompare_chocosim");
    fs::create_directories("HomingSimulationCompare_chocosim");

    // Generate or read seeds
    std::vector<std::string> seeds;
    if (argc == 1) {
        // Generate random seeds
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 999999);
        for (int i = 0; i < 10; ++i) {
            seeds.push_back(std::to_string(distrib(gen)));
        }
        
        // Save seeds to log file
        std::ofstream log_seeds("HomingSimulationCompare_chocosim/log_seeds.txt");
        for (const auto& seed : seeds) {
            log_seeds << seed << "\n";
        }
    } else if (argc == 2) {
        // Read seeds from file
        std::ifstream seeds_file(argv[1]);
        std::string seed;
        while (std::getline(seeds_file, seed)) {
            seeds.push_back(seed);
        }
    } else {
        std::cerr << "Cannot process more than one argument" << std::endl;
        return 1;
    }

    // Set up OpenMP threads
    int num_threads = std::thread::hardware_concurrency();
    omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " threads." << std::endl;

    // Run simulations for each combination
    for (const auto& mission : missions) {
        for (const auto& method : methods) {
            std::string cs_dir = "HomingDesign/" + mission + "/" + method;
            if (mission == "Homing" && method == "Chocolate") {
                cs_dir = "Design/" + mission + "/" + method;
            }

            std::string argos_file_name = "argos/homing_auto.argos";

            for (const auto& environment : environments) {
                std::cout << "Running simulations for " << mission << " - " << method << " - " << environment << std::endl;
                std::cout << "Using ARGoS file: " << argos_file_name << std::endl;
                std::cout << "Using cs_dir: " << cs_dir << std::endl;

                runSimulations(mission, method, environment, seeds, cs_dir, argos_file_name);
            }
        }
    }

    return 0;
}
