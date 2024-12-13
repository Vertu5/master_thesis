#include <iostream>
#include "SimulationProcessor.h"

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [mission_name]" << std::endl;
    std::cout << "If no mission name is provided, all missions will be processed." << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        SimulationProcessor processor("SimulationResults_Compare", "Persistence_analysis");

        if (argc > 1) {
            // Traiter une mission spécifique
            std::string mission_name = argv[1];
            processor.processMission(mission_name);
        }
        else {
            // Traiter toutes les missions
            processor.processAllMissions();
        }

        std::cout << "\nProcessing complete!" << std::endl;
        return 0;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
