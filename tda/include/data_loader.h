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

class DataLoader {
public:
    DataLoader(const std::string& base_directory);

    // Main loading function
    std::vector<std::vector<RunData>> loadAllMissions();
    
    // Save/Load cache functions
    bool saveToFile(const std::vector<std::vector<RunData>>& data, const std::string& filename) const;
    std::vector<std::vector<RunData>> loadFromFile(const std::string& filename) const;

private:
    std::string base_directory_;
    
    // Helper functions from original implementation
    bool readBestFSMs(std::vector<std::string>& missions, std::vector<std::string>& best_fsms) const;
    int findFSMNumber(const std::string& best_fsm, const std::string& autocontrollers_path) const;
    bool readPerformanceData(const std::string& mission_name, int fsm_number, std::vector<RunData>& data) const;
    
    // Cache management
    std::string getBinaryFilePath() const;
    bool fileExists(const std::string& filename) const;
};

#endif // DATA_LOADER_H