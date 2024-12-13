#include <iostream>
#include <string>
#include <filesystem>
#include "silhouette_analyzer.h"

namespace fs = std::filesystem;

void analyze_two_runs(const std::string& run_file1, const std::string& run_file2) {
    try {
        SilhouetteAnalyzer analyzer;
        
        std::cout << "Chargement des données des runs...\n";
        
        std::cout << "Calcul des silhouettes...\n";
        auto silhouettes1 = analyzer.compute_all_silhouettes(run_file1);
        auto silhouettes2 = analyzer.compute_all_silhouettes(run_file2);
        
        // Extract run names from file paths
        fs::path path1(run_file1);
        fs::path path2(run_file2);
        std::string run1_name = path1.stem().string();
        std::string run2_name = path2.stem().string();
        
        // Extract run numbers (assuming format like "run_89")
        size_t pos1 = run1_name.find("run_");
        size_t pos2 = run2_name.find("run_");
        run1_name = pos1 != std::string::npos ? run1_name.substr(pos1) : run1_name;
        run2_name = pos2 != std::string::npos ? run2_name.substr(pos2) : run2_name;
        
        std::cout << "Alignement des silhouettes...\n";
        auto [aligned1, aligned2] = analyzer.align_silhouettes(silhouettes1, silhouettes2);
        
        // Save aligned silhouettes
        for (const auto& [field_type, field_silhouettes] : aligned1) {
            for (const auto& [diagram_type, silhouette] : field_silhouettes) {
                std::string filename = "aligned_" + field_type + "_" + diagram_type + "_" + run1_name + ".csv";
                save_silhouette(filename, silhouette);
                std::cout << "Sauvegardé " << filename << std::endl;
            }
        }
        
        for (const auto& [field_type, field_silhouettes] : aligned2) {
            for (const auto& [diagram_type, silhouette] : field_silhouettes) {
                std::string filename = "aligned_" + field_type + "_" + diagram_type + "_" + run2_name + ".csv";
                save_silhouette(filename, silhouette);
                std::cout << "Sauvegardé " << filename << std::endl;
            }
        }
        
        std::cout << "Analyse terminée!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors de l'analyse: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <run_file1> <run_file2>\n";
        std::cerr << "Example: " << argv[0] 
                 << " output/persistence/unweighted/runs/Foraging_Chocolate_simulation_run_89_persistence.csv"
                 << " output/persistence/unweighted/runs/Foraging_Chocolate_simulation_run_19_persistence.csv\n";
        return 1;
    }
    
    std::string run_file1 = argv[1];
    std::string run_file2 = argv[2];
    
    analyze_two_runs(run_file1, run_file2);
    return 0;
}
