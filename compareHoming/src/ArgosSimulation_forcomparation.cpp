#include "ArgosSimulation_forcomparation.h"
#include <iostream>
#include <fstream>
#include <random>
#include <filesystem>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <cctype>
#include <omp.h>
#include <mutex>
#include <cmath>

namespace fs = std::filesystem;

ArgosFile::ArgosFile(const std::string& name) : name(name) {
    pugi::xml_parse_result result = doc.load_file(name.c_str());
    if (!result) {
        throw std::runtime_error("Failed to load .argos file: " + name);
    }
    root = doc.root();
}

void ArgosFile::setDuration(const std::string& duration) {
    auto experiment = root.select_node("//experiment").node();
    if (experiment) {
        experiment.attribute("length") = duration.c_str();
    }
}

void ArgosFile::setSeed(const std::string& seed) {
    auto experiment = root.select_node("//experiment").node();
    if (experiment) {
        experiment.attribute("random_seed") = seed.c_str();
    }
}

void ArgosFile::setLog(const std::string& log) {
    auto params = root.select_node("//loop_functions/params").node();
    if (params) {
        params.attribute("log_filename") = log.c_str();
    } else {
        std::cerr << "Could not find //loop_functions/params node in XML." << std::endl;
    }
}

void ArgosFile::setFsm(const std::string& fsm) {
    auto params = root.select_node("//automode_controller/params").node();
    if (params) {
        params.attribute("fsm-config") = fsm.c_str();
    }
}

void ArgosFile::setGenome(const std::string& genome) {
    auto params = root.select_node("//nn_rm_1dot1_controller/params").node();
    if (params) {
        params.attribute("genome_file") = genome.c_str();
    }
}

void ArgosFile::setSim() {
    auto light = root.select_node("//epuck_light").node();
    if (light) {
        light.attribute("noise_level") = "0.05";
    }
    auto rnb = root.select_node("//sensors/epuck_range_and_bearing").node();
    if (rnb) {
        rnb.attribute("loss_probability") = "0.85";
    }
    auto wheels = root.select_node("//epuck_wheels").node();
    if (wheels) {
        wheels.attribute("noise_std_dev") = "0.05";
    }
}

void ArgosFile::setPr() {
    auto light = root.select_node("//epuck_light").node();
    if (light) {
        light.attribute("noise_level") = "0.9";
    }
    auto rnb = root.select_node("//sensors/epuck_range_and_bearing").node();
    if (rnb) {
        rnb.attribute("loss_probability") = "0.9";
    }
    auto wheels = root.select_node("//epuck_wheels").node();
    if (wheels) {
        wheels.attribute("noise_std_dev") = "0.15";
    }
}

void ArgosFile::run() {
    std::string tmp_filename = "tmp_argos_" + std::to_string(rand()) + ".argos";
    doc.save_file(tmp_filename.c_str());
    std::string command = "argos3 -c " + tmp_filename + " -n > /dev/null";

    int ret = std::system(command.c_str());
    if (ret != 0) {
        std::cerr << "Error running ARGoS simulation with command: " << command << std::endl;
    }

    fs::remove(tmp_filename);
}

std::vector<std::string> extractFSM(const std::string& cs_dir) {
    std::vector<std::string> fsms;
    for (const auto& entry : fs::directory_iterator(cs_dir)) {
        if (entry.is_regular_file()) {
            std::ifstream file(entry.path());
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) {
                    fsms.push_back(line);
                }
            }
        }
    }
    return fsms;
}

std::string to_lowercase(const std::string& str) {
    std::string lower_str;
    lower_str.resize(str.size());
    std::transform(str.begin(), str.end(), lower_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return lower_str;
}

std::pair<std::string, double> runSimulations(const std::string& mission, const std::string& method, const std::string& environment,
                    const std::vector<std::string>& seeds, const std::string& cs_dir, const std::string& argos_file_name) {
    int total_simulations = 0;
    int completed_simulations = 0;
    std::string best_control_software;
    double best_performance = 0.0;  // Changed to 0.0 since we're not using it
    std::mutex mtx;

    if (method == "Chocolate") {
        std::vector<std::string> fsm_list = extractFSM(cs_dir);
        if (fsm_list.empty()) {
            std::cerr << "No FSMs found in directory: " << cs_dir << std::endl;
            return {best_control_software, best_performance};
        }
        total_simulations = fsm_list.size() * seeds.size();

        #pragma omp parallel
        {
            ArgosFile argos_file(argos_file_name);
            if (environment == "simulation") {
                argos_file.setSim();
            } else if (environment == "pseudoreality") {
                argos_file.setPr();
            }

            #pragma omp for schedule(dynamic)
            for (size_t n = 0; n < fsm_list.size(); ++n) {
                std::string log_dir = "HomingSimulationCompare_chocosim/" + mission + "/" + method + "/" + environment + "/" + std::to_string(n);
                fs::create_directories(log_dir);
                argos_file.setFsm(fsm_list[n]);

                for (const auto& seed : seeds) {
                    std::string log_filename = log_dir + "/" + seed + ".csv";
                    argos_file.setLog(log_filename);
                    argos_file.setSeed(seed);
                    argos_file.run();

                    #pragma omp critical
                    {
                        completed_simulations++;
                        float progress = (float)completed_simulations / total_simulations * 100.0f;
                        std::cout << "\rProgress: " << std::fixed << std::setprecision(1) << progress << "% ("
                                  << completed_simulations << "/" << total_simulations << ")" << std::flush;
                    }
                }
            }
        }
    }
    std::cout << std::endl;
    
    return {best_control_software, best_performance};
}