#include "ArgosSimulation.h"
#include <iostream>
#include <fstream>
#include <random>
#include <filesystem>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <cctype>
#include <omp.h>
#include <limits>
#include <mutex>
#include <numeric>
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

double readPerformanceFromLog(const std::string& log_filename) {
    std::ifstream log_file(log_filename);
    std::string line;
    double performance = 0.0;
    
    while (std::getline(log_file, line)) {
        try {
            performance = std::stod(line);
        } catch (const std::exception& e) {
            // If conversion fails, continue to the next line
            continue;
        }
    }
    
    return performance;
}

std::pair<std::string, double> runSimulations(const std::string& mission, const std::string& method, const std::string& environment,
                    const std::vector<std::string>& seeds, const std::string& cs_dir, const std::string& argos_file_name) {
    int total_simulations = 0;
    int completed_simulations = 0;
    std::string best_control_software;
    double best_performance = -std::numeric_limits<double>::infinity();

    std::mutex mtx; // Mutex for thread-safe operations
    std::vector<double> average_performances;
    std::vector<std::string> control_softwares;
    std::vector<std::vector<double>> all_performances; // Stores all performances for each FSM

    if (method == "Chocolate") {
        std::vector<std::string> fsm_list = extractFSM(cs_dir);
        if (fsm_list.empty()) {
            std::cerr << "No FSMs found in directory: " << cs_dir << std::endl;
            return {best_control_software, best_performance};
        }
        total_simulations = fsm_list.size() * seeds.size();
        average_performances.resize(fsm_list.size(), 0.0);
        control_softwares = fsm_list;
        all_performances.resize(fsm_list.size());

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
                std::string log_dir = "argoslogs_chocosim/" + mission + "/" + method + "/" + environment + "/" + std::to_string(n);
                fs::create_directories(log_dir);
                argos_file.setFsm(fsm_list[n]);

                double total_performance = 0.0;
                std::vector<double> performances;

                for (const auto& seed : seeds) {
                    std::string log_filename = log_dir + "/" + seed + ".csv";
                    argos_file.setLog(log_filename);
                    argos_file.setSeed(seed);
                    argos_file.run();

                    double performance = readPerformanceFromLog(log_filename);
                    total_performance += performance;
                    performances.push_back(performance);

                    #pragma omp critical
                    {
                        completed_simulations++;
                        float progress = (float)completed_simulations / total_simulations * 100.0f;
                        std::cout << "\rProgress: " << std::fixed << std::setprecision(1) << progress << "% ("
                                  << completed_simulations << "/" << total_simulations << ")" << std::flush;
                    }
                }

                double average_performance = total_performance / seeds.size();
                average_performances[n] = average_performance;
                all_performances[n] = performances;

                #pragma omp critical
                {
                    if (average_performance > best_performance) {
                        best_performance = average_performance;
                        best_control_software = fsm_list[n];
                    }
                }
            }
        }

        // Save the performance data to a CSV file
        std::string csv_filename = "performance_data_" + mission + "_" + method + "_" + environment + ".csv";
        std::ofstream csv_file(csv_filename);
        csv_file << "FSM,Mean,StdDev,Variance,Performances\n";
        for (size_t i = 0; i < fsm_list.size(); ++i) {
            // Calculate statistical metrics
            double mean = average_performances[i];
            double variance = 0.0;
            for (const auto& perf : all_performances[i]) {
                variance += (perf - mean) * (perf - mean);
            }
            variance /= seeds.size();
            double stddev = std::sqrt(variance);

            // Write data to CSV
            csv_file << "\"" << fsm_list[i] << "\"," << mean << "," << stddev << "," << variance << ",\"";
            for (size_t j = 0; j < all_performances[i].size(); ++j) {
                csv_file << all_performances[i][j];
                if (j != all_performances[i].size() - 1) {
                    csv_file << ";";
                }
            }
            csv_file << "\"\n";
        }
        csv_file.close();

    } else if (method == "Evostick") {
        // ... (Evostick implementation remains the same, but add similar logic for collecting data)
    }
    std::cout << std::endl;
    
    return {best_control_software, best_performance};
}
