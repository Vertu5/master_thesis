#ifndef SILHOUETTE_UTILS_H
#define SILHOUETTE_UTILS_H

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

inline void save_silhouette(const std::string& filename, const std::vector<double>& silhouette) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Impossible d'ouvrir le fichier pour écriture: " + filename);
    }

    file << "value\n";  // En-tête
    for (double value : silhouette) {
        file << value << "\n";
    }
}

#endif // SILHOUETTE_UTILS_H
