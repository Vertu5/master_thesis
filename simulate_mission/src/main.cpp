#include "ArgosSimulation.h"
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
    std::vector<std::string> missions = {"DirectionalGate", "Foraging", "Homing", "Shelter", "XOR-Aggregation"};
    std::map<std::string, std::string> mission_lowercase_map = {
        {"DirectionalGate", "directionalgate"},
        {"Foraging", "foraging"},
        {"Homing", "homing"},
        {"Shelter", "shelter"},
        {"XOR-Aggregation", "xor_aggregation"}
    };
    std::vector<std::string> methods = {"Chocolate"};
    std::vector<std::string> environments = {"simulation"};

    fs::remove_all("argoslogs_chocosim");
    fs::create_directories("argoslogs_chocosim");

    std::vector<std::string> seeds;
    if (argc == 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 999999);
        for (int i = 0; i < 200; ++i) {
            seeds.push_back(std::to_string(distrib(gen)));
        }
        std::ofstream log_seeds("argoslogs_chocosim/log_seeds.txt");
        for (const auto& seed : seeds) {
            log_seeds << seed << "\n";
        }
    } else if (argc == 2) {
        std::ifstream seeds_file(argv[1]);
        std::string seed;
        while (std::getline(seeds_file, seed)) {
            seeds.push_back(seed);
        }
    } else {
        std::cerr << "Cannot process more than one argument" << std::endl;
        return 1;
    }

    int num_threads = std::thread::hardware_concurrency();
    omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " threads." << std::endl;

    std::map<std::string, std::pair<std::string, double>> best_control_softwares;

    for (const auto& mission : missions) {
        for (const auto& method : methods) {
            std::string cs_dir = "Design/" + mission + "/" + method;

            std::map<std::string, std::string> method_suffixes = {
                {"Chocolate", "auto"},
                {"Evostick", "evostick"}
            };
            std::string argos_file_name = "argos/" + mission_lowercase_map[mission] + "_" + method_suffixes[method] + ".argos";

            for (const auto& environment : environments) {
                std::cout << "Running simulations for " << mission << " - " << method << " - " << environment << std::endl;
                std::cout << "Using ARGoS file: " << argos_file_name << std::endl;
                std::cout << "Using cs_dir: " << cs_dir << std::endl;

                auto [best_cs, performance] = runSimulations(mission, method, environment, seeds, cs_dir, argos_file_name);
                
                std::string key = mission + "_" + method + "_" + environment;
                best_control_softwares[key] = {best_cs, performance};
            }
        }
    }

    // Print the best control software for each combination
    for (const auto& [key, value] : best_control_softwares) {
        std::cout << "Best Control Software for " << key << ":" << std::endl;
        std::cout << "FSM/Genome: " << value.first << std::endl;
        std::cout << "Performance: " << value.second << std::endl;
        std::cout << std::endl;
    }
    return 0;
}
