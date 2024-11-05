#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>

struct RunData {
    std::vector<std::vector<double>> x;  // Positions x[t][i]
    std::vector<std::vector<double>> y;  // Positions y[t][i]
    std::vector<std::vector<double>> vx; // Velocities vx[t][i]
    std::vector<std::vector<double>> vy; // Velocities vy[t][i]
    double objective_function;
    size_t num_robots;
    size_t num_time_steps;
};

bool read_best_fsms(const std::string& file_path, std::vector<std::string>& missions, std::vector<std::string>& best_fsms);

int find_fsm_number(const std::string& fsm, const std::string& autocontrollers_path);

bool read_performance_data(const std::string& mission_name, int fsm_number, std::vector<RunData>& all_data);

#endif // DATA_LOADER_H
