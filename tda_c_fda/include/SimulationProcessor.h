#ifndef SIMULATION_PROCESSOR_H
#define SIMULATION_PROCESSOR_H

#include <string>
#include <vector>
#include <filesystem>
#include <optional>
#include <functional>
#include "data_loader.h"
#include "RGrid.h"
#include "VectorField.h"
#include "HHD.h"
#include "ExtendedPersistence.h"

namespace fs = std::filesystem;

struct OutputPaths {
    fs::path base_dir;
    fs::path unweighted_missions;
    fs::path unweighted_runs;
    fs::path weighted_missions;
    fs::path weighted_runs;

    OutputPaths(const std::string& base = "Persistence_analysis") : base_dir(base) {
        unweighted_missions = base_dir / "persistence" / "unweighted" / "missions";
        unweighted_runs = base_dir / "persistence" / "unweighted" / "runs";
        weighted_missions = base_dir / "persistence" / "weighted" / "missions";
        weighted_runs = base_dir / "persistence" / "weighted" / "runs";

        for(const auto& dir : {
            base_dir,
            base_dir / "persistence",
            base_dir / "persistence" / "unweighted",
            base_dir / "persistence" / "weighted",
            unweighted_missions,
            unweighted_runs,
            weighted_missions,
            weighted_runs
        }) {
            std::filesystem::create_directories(dir);
        }
    }
};

class SimulationProcessor {
public:
    SimulationProcessor(const std::string& simulation_dir = "SimulationResults_Compare",
                       const std::string& output_dir = "Persistence_analysis");

    void processAllMissions();
    void processMission(const std::string& mission_name);

private:
    const double optimal_w = 0.2043;
    const double arena_size = 2 * 1.231;
    fs::path simulation_dir_;
    OutputPaths output_paths_;

    std::vector<std::string> findAvailableMissions() const;
    RunData readSimulationCSV(const fs::path& file_path);
    std::vector<RunData> readPFSMDirectory(const fs::path& pfsm_path);
    void computeAndSavePersistence(const RunData& data,
                                 const std::string& mission_name,
                                 const std::string& type,
                                 int pfsm_index = -1,
                                 int run_index = -1);
};

#endif // SIMULATION_PROCESSOR_H
