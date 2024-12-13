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

/**
 * @class DataLoader
 * @brief A class responsible for loading and caching mission data.
 *
 * The DataLoader class provides functionality to load mission data from a specified base directory,
 * save and load cached data to and from files, and manage the cache.
 */
 
/**
 * @brief Constructs a DataLoader object with the specified base directory.
 * @param base_directory The base directory where mission data is located.
 */
 
/**
 * @brief Loads all missions and their associated data.
 * @return A pair containing a vector of vectors of RunData and a vector of mission names.
 */
 
/**
 * @brief Saves the provided data to a file.
 * @param data The data to be saved.
 * @param filename The name of the file to save the data to.
 * @return True if the data was successfully saved, false otherwise.
 */
 
/**
 * @brief Loads data from a specified file.
 * @param filename The name of the file to load the data from.
 * @return A vector of vectors of RunData loaded from the file.
 */
 
/**
 * @brief Reads the best FSMs (Finite State Machines) for the missions.
 * @param missions A reference to a vector of mission names to be populated.
 * @param best_fsms A reference to a vector of best FSMs to be populated.
 * @return True if the best FSMs were successfully read, false otherwise.
 */
 
/**
 * @brief Finds the FSM number for a given best FSM and autocontrollers path.
 * @param best_fsm The best FSM name.
 * @param autocontrollers_path The path to the autocontrollers.
 * @return The FSM number.
 */
 
/**
 * @brief Reads performance data for a given mission and FSM number.
 * @param mission_name The name of the mission.
 * @param fsm_number The FSM number.
 * @param data A reference to a vector of RunData to be populated.
 * @return True if the performance data was successfully read, false otherwise.
 */
 
/**
 * @brief Gets the file path for the binary cache file.
 * @return The binary file path as a string.
 */
 
/**
 * @brief Checks if a file exists.
 * @param filename The name of the file to check.
 * @return True if the file exists, false otherwise.
 */
class DataLoader {
public:
    DataLoader(const std::string& base_directory);

    // Main loading function
    std::pair<std::vector<std::vector<RunData>>, std::vector<std::string>> loadAllMissions();
    
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