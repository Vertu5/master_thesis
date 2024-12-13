#ifndef SILHOUETTE_ANALYZER_H
#define SILHOUETTE_ANALYZER_H

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include "Silhouette.h"
#include "silhouette_utils.h"
#include "fdacurve.hpp"

struct CSVRow {
    std::string type;
    std::string diagram;
    std::string field;
    double birth;
    double death;
};

struct DiagramType {
    std::string type;
    std::string dimension;
    
    DiagramType(const std::string& t, const std::string& d) 
        : type(t), dimension(d) {}
};

class SilhouetteAnalyzer {
public:
    SilhouetteAnalyzer() : resolution_(200) {
        velocity_fields_ = {"velocity_divergent", "velocity_rotational"};
        diagram_types_ = {
            DiagramType("ordinary", "h0"),
            DiagramType("ordinary", "h1"),
            DiagramType("relative", "h1"),
            DiagramType("relative", "h2"),
            DiagramType("extended_plus", "h0"),
            DiagramType("extended_minus", "h1")
        };
    }

    std::vector<double> compute_single_silhouette(
        const std::vector<std::pair<double, double>>& points) {
        
        if (points.empty()) {
            return std::vector<double>(resolution_, 0.0);
        }

        // Filtrer les points valides
        std::vector<std::pair<double, double>> valid_points;
        for (const auto& point : points) {
            if (std::isfinite(point.first) && std::isfinite(point.second) &&
                point.first != point.second) {
                valid_points.push_back(point);
            }
        }

        if (valid_points.empty()) {
            return std::vector<double>(resolution_, 0.0);
        }

        std::vector<std::vector<std::pair<double, double>>> diagrams = {valid_points};
        
        PersistenceSilhouette silhouette(
            [](const std::pair<double, double>&) { return 1.0; },
            resolution_
        );

        return silhouette.transform(diagrams)[0];
    }

    using SilhouetteMap = std::map<std::string, std::map<std::string, std::vector<double>>>;

    SilhouetteMap compute_all_silhouettes(const std::string& csv_file) {
        SilhouetteMap silhouettes;
        auto data = load_csv(csv_file);

        for (const auto& field_type : velocity_fields_) {
            auto field_data = filter_by_field(data, field_type);
            
            for (const auto& diagram_type : diagram_types_) {
                auto type_data = filter_by_type(field_data, diagram_type);
                std::string key = diagram_type.type + "_" + diagram_type.dimension;
                silhouettes[field_type][key] = 
                    compute_single_silhouette(extract_birth_death_pairs(type_data));
            }
        }

        return silhouettes;
    }

    std::pair<SilhouetteMap, SilhouetteMap> align_silhouettes(
        const SilhouetteMap& silhouettes1,
        const SilhouetteMap& silhouettes2) {
        
        try {
            size_t total_diagrams = diagram_types_.size() * velocity_fields_.size();
            std::cout << "Nombre total de diagrammes par run: " << total_diagrams << std::endl;
            
            // Créer deux matrices beta séparées, une pour chaque run
            Eigen::MatrixXd beta1 = Eigen::MatrixXd::Zero(total_diagrams, resolution_);
            Eigen::MatrixXd beta2 = Eigen::MatrixXd::Zero(total_diagrams, resolution_);
            std::cout << "Taille de chaque matrice beta: " << beta1.rows() << "x" << beta1.cols() << std::endl;
            
            size_t idx = 0;
            for (const auto& field_type : velocity_fields_) {
                std::cout << "\nTraitement du champ " << field_type << std::endl;
                for (const auto& diagram_type : diagram_types_) {
                    std::string key = diagram_type.type + "_" + diagram_type.dimension;
                    std::cout << "  Diagramme " << key << ":" << std::endl;
                    
                    // Premier run
                    const auto& sil1 = silhouettes1.at(field_type).at(key);
                    beta1.row(idx) = Eigen::Map<const Eigen::VectorXd>(sil1.data(), sil1.size());
                    
                    double min1 = *std::min_element(sil1.begin(), sil1.end());
                    double max1 = *std::max_element(sil1.begin(), sil1.end());
                    double sum1 = std::accumulate(sil1.begin(), sil1.end(), 0.0);
                    std::cout << "    Run1 - Min: " << min1 << ", Max: " << max1 << ", Sum: " << sum1 << std::endl;
                    
                    // Deuxième run
                    const auto& sil2 = silhouettes2.at(field_type).at(key);
                    beta2.row(idx) = Eigen::Map<const Eigen::VectorXd>(sil2.data(), sil2.size());
                    
                    double min2 = *std::min_element(sil2.begin(), sil2.end());
                    double max2 = *std::max_element(sil2.begin(), sil2.end());
                    double sum2 = std::accumulate(sil2.begin(), sil2.end(), 0.0);
                    std::cout << "    Run2 - Min: " << min2 << ", Max: " << max2 << ", Sum: " << sum2 << std::endl;
                    
                    idx++;
                }
            }
            
            // Vérifier si toutes les silhouettes sont nulles
            bool all_zeros = (beta1.array().abs().sum() < 1e-10) && (beta2.array().abs().sum() < 1e-10);
            if(all_zeros) {
                std::cout << "ATTENTION: Toutes les silhouettes sont nulles!" << std::endl;
                return {silhouettes1, silhouettes2};
            }
            
            std::cout << "\nPréparation de l'alignement..." << std::endl;
            
            // Créer un vecteur contenant les deux matrices
            std::vector<Eigen::MatrixXd> beta_input = {beta1, beta2};
            
            std::cout << "Création de FDACurve avec " << beta_input.size() << " courbes..." << std::endl;
            FDACurve fdacurve(beta_input, "O", resolution_, false);
            
            std::cout << "Calcul de la moyenne de Karcher..." << std::endl;
            fdacurve.karcher_mean(false);
            
            std::cout << "Alignement SRVF..." << std::endl;
            fdacurve.srvf_align(false);
            
            // Récupérer les résultats alignés
            const auto& aligned_curves = fdacurve.get_betan();
            if (aligned_curves.size() < 2) {
                throw std::runtime_error("Nombre insuffisant de courbes alignées retournées");
            }
            
            std::cout << "Nombre de courbes alignées: " << aligned_curves.size() << std::endl;
            std::cout << "Dimensions des courbes alignées: " << aligned_curves[0].rows() << "x" << 
                    aligned_curves[0].cols() << std::endl;
            
            SilhouetteMap aligned1 = silhouettes1;
            SilhouetteMap aligned2 = silhouettes2;
            
            idx = 0;
            for (const auto& field_type : velocity_fields_) {
                for (const auto& diagram_type : diagram_types_) {
                    std::string key = diagram_type.type + "_" + diagram_type.dimension;
                    
                    // Utiliser les courbes alignées correspondantes
                    aligned1[field_type][key] = std::vector<double>(
                        aligned_curves[0].row(idx).data(),
                        aligned_curves[0].row(idx).data() + resolution_
                    );
                    
                    aligned2[field_type][key] = std::vector<double>(
                        aligned_curves[1].row(idx).data(),
                        aligned_curves[1].row(idx).data() + resolution_
                    );
                    
                    idx++;
                }
            }
            
            std::cout << "Alignement terminé avec succès" << std::endl;
            return {aligned1, aligned2};
            
        } catch (const std::exception& e) {
            std::cerr << "Erreur lors de l'alignement: " << e.what() << std::endl;
            return {silhouettes1, silhouettes2};
        }
    }
    
private:
    size_t resolution_;
    std::vector<std::string> velocity_fields_;
    std::vector<DiagramType> diagram_types_;

    std::vector<CSVRow> load_csv(const std::string& filename) {
        std::vector<CSVRow> data;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            throw std::runtime_error("Impossible d'ouvrir le fichier: " + filename);
        }

        std::string line;
        // Skip header
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::vector<std::string> cells;
            std::stringstream ss(line);
            std::string cell;
            
            while (std::getline(ss, cell, ',')) {
                cells.push_back(cell);
            }
            
            if (cells.size() == 5) {
                CSVRow row;
                row.type = cells[0];
                row.diagram = cells[1];
                row.field = cells[2];
                try {
                    row.birth = std::stod(cells[3]);
                    row.death = std::stod(cells[4]);
                    data.push_back(row);
                } catch (const std::exception& e) {
                    continue;
                }
            }
        }

        return data;
    }

    std::vector<CSVRow> filter_by_field(
        const std::vector<CSVRow>& data, 
        const std::string& field_type) {
        
        std::vector<CSVRow> filtered_data;
        for (const auto& row : data) {
            if (row.field == field_type) {
                filtered_data.push_back(row);
            }
        }
        return filtered_data;
    }

    std::vector<CSVRow> filter_by_type(
        const std::vector<CSVRow>& data, 
        const DiagramType& diagram_type) {
        
        std::vector<CSVRow> filtered_data;
        for (const auto& row : data) {
            if (row.type == diagram_type.type && 
                row.diagram == diagram_type.dimension) {
                filtered_data.push_back(row);
            }
        }
        return filtered_data;
    }

    std::vector<std::pair<double, double>> extract_birth_death_pairs(
        const std::vector<CSVRow>& data) {
        
        std::vector<std::pair<double, double>> pairs;
        for (const auto& row : data) {
            pairs.emplace_back(row.birth, row.death);
        }
        return pairs;
    }
};

#endif // SILHOUETTE_ANALYZER_H
