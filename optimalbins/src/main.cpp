#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>  // Added for setprecision
#include "data_loader.h"
#include "bin_width_optimizer.h"

// Helper function to calculate mean of a vector
template<typename T>
double calculate_mean(const std::vector<T>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

int main() {
    // Load best FSMs
    std::vector<std::string> missions;
    std::vector<std::string> best_fsms;
    std::cout << "Reading best FSMs..." << std::endl;
    if (!read_best_fsms("best_fsms.csv", missions, best_fsms)) {
        return 1;
    }

    // Collect all mission data
    std::vector<std::vector<RunData>> all_mission_data;
    for (size_t idx = 0; idx < missions.size(); ++idx) {
        std::cout << "Processing mission " << missions[idx] << "..." << std::endl;
        const std::string& mission = missions[idx];
        const std::string& best_fsm = best_fsms[idx];
        std::string mission_name = mission.substr(0, mission.find('_'));
        std::string autocontrollers_path = "Design/" + mission_name + "/Chocolate/autocontrollers.txt";
        
        std::cout << "Reading performance data for " << mission_name << "..." << std::endl;
        int fsm_number = find_fsm_number(best_fsm, autocontrollers_path);
        if (fsm_number != -1) {
            std::vector<RunData> mission_data;
            if (read_performance_data(mission_name, fsm_number, mission_data)) {
                all_mission_data.push_back(mission_data);
                
                // Calculate averages for this mission
                std::vector<double> objective_functions;
                std::vector<int> num_robots;
                
                for (const auto& run : mission_data) {
                    objective_functions.push_back(run.objective_function);
                    num_robots.push_back(run.num_robots);
                }
                
                double avg_performance = calculate_mean(objective_functions);
                double avg_num_robots = calculate_mean(num_robots);
                
                std::cout << "\nMission: " << mission << std::endl;
                std::cout << "Average Performance: " << std::fixed << std::setprecision(4) 
                         << avg_performance << std::endl;
                std::cout << "Average Number of Robots: " << std::fixed << std::setprecision(2) 
                         << avg_num_robots << std::endl;
            } else {
                std::cerr << "No valid data found for " << mission_name << std::endl;
            }
        } else {
            std::cerr << "Could not find FSM number for " << mission_name << std::endl;
        }
    }

    // Find optimal bin width
    double optimal_w = 0.0;
    std::vector<double> w_range;
    std::vector<double> quality_scores;
    if (find_optimal_bin_width(all_mission_data, optimal_w, w_range, quality_scores)) {
        std::cout << "Optimal bin width: " << optimal_w << " m" << std::endl;
    } else {
        std::cerr << "Failed to find optimal bin width." << std::endl;
        return 1;
    }

    // Display quality scores
    for (size_t i = 0; i < w_range.size(); ++i) {
        std::cout << "Bin width: " << w_range[i] << ", Quality score: " << quality_scores[i] << std::endl;
    }

    return 0;
}