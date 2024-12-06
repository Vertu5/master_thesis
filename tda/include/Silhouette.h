#ifndef PERSISTENCE_SILHOUETTE_H
#define PERSISTENCE_SILHOUETTE_H

#include <vector>
#include <functional>
#include <limits>
#include <cmath>
#include <algorithm>
#include <optional>

using PersistenceDiagram = std::vector<std::pair<double, double>>;

class PersistenceSilhouette {
public:
    // Type pour la fonction de poids
    using WeightFunction = std::function<double(const std::pair<double, double>&)>;

    struct Range {
        double min = std::numeric_limits<double>::quiet_NaN();
        double max = std::numeric_limits<double>::quiet_NaN();
        
        bool is_valid() const {
            return !std::isnan(min) && !std::isnan(max);
        }
    };

    // Constructeur avec paramètres par défaut
    PersistenceSilhouette(
        WeightFunction weight = [](const std::pair<double, double>& p) { return 1.0; },
        size_t resolution = 100,
        Range sample_range = Range{},
        bool keep_endpoints = false
    ) : weight_(weight), 
        resolution_(resolution), 
        sample_range_(sample_range),
        keep_endpoints_(keep_endpoints),
        grid_(resolution) {}

    // Calcule le range à partir des diagrammes si nécessaire
    void fit(const std::vector<PersistenceDiagram>& diagrams) {
        if (!sample_range_.is_valid()) {
            compute_sample_range(diagrams);
        }
        compute_grid();
    }

    // Transforme un ensemble de diagrammes en silhouettes
    std::vector<std::vector<double>> transform(
        const std::vector<PersistenceDiagram>& diagrams) {
        if (!sample_range_.is_valid()) {
            fit(diagrams);
        }

        std::vector<std::vector<double>> result;
        result.reserve(diagrams.size());
        for (const auto& diagram : diagrams) {
            result.push_back(compute_silhouette(diagram));
        }
        return result;
    }

    // Transforme un seul diagramme en silhouette
    std::vector<double> fit_transform(const PersistenceDiagram& diagram) {
        std::vector<PersistenceDiagram> diagrams{diagram};
        if (!sample_range_.is_valid()) {
            fit(diagrams);
        }
        return compute_silhouette(diagram);
    }

    // Accesseurs
    const std::vector<double>& grid() const { return grid_; }
    Range sample_range() const { return sample_range_; }
    size_t resolution() const { return resolution_; }

private:
    WeightFunction weight_;
    size_t resolution_;
    Range sample_range_;
    bool keep_endpoints_;
    std::vector<double> grid_;

    void compute_sample_range(const std::vector<PersistenceDiagram>& diagrams) {
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();

        for (const auto& diagram : diagrams) {
            for (const auto& point : diagram) {
                double midpoint = (point.first + point.second) / 2.0;
                double height = (point.second - point.first) / 2.0;
                
                min_val = std::min(min_val, midpoint - height);
                max_val = std::max(max_val, midpoint + height);
            }
        }

        if (!keep_endpoints_) {
            // Ajuster légèrement le range pour éviter les extrémités exactes
            double range = max_val - min_val;
            min_val += range * 0.02;
            max_val -= range * 0.02;
        }

        sample_range_.min = min_val;
        sample_range_.max = max_val;
    }

    void compute_grid() {
        if (!sample_range_.is_valid()) {
            throw std::runtime_error("Sample range not computed yet");
        }

        grid_.resize(resolution_);
        double step = (sample_range_.max - sample_range_.min) / (resolution_ - 1);
        for (size_t i = 0; i < resolution_; ++i) {
            grid_[i] = sample_range_.min + i * step;
        }
    }

    std::vector<double> compute_silhouette(const PersistenceDiagram& diagram) {
        if (diagram.empty()) {
            return std::vector<double>(resolution_, 0.0);
        }

        // Calcul des poids totaux
        std::vector<double> weights;
        weights.reserve(diagram.size());
        double total_weight = 0.0;
        
        for (const auto& point : diagram) {
            double w = weight_(point);
            weights.push_back(w);
            total_weight += w;
        }

        // Calcul des midpoints et heights
        std::vector<double> midpoints(diagram.size());
        std::vector<double> heights(diagram.size());
        
        for (size_t i = 0; i < diagram.size(); ++i) {
            midpoints[i] = (diagram[i].first + diagram[i].second) / 2.0;
            heights[i] = (diagram[i].second - diagram[i].first) / 2.0;
        }

        // Calcul de la silhouette
        std::vector<double> silhouette(resolution_, 0.0);
        
        for (size_t i = 0; i < resolution_; ++i) {
            double x = grid_[i];
            for (size_t j = 0; j < diagram.size(); ++j) {
                silhouette[i] += (weights[j] / total_weight) * 
                                std::max(heights[j] - std::abs(x - midpoints[j]), 0.0);
            }
        }

        // Multiplication finale par sqrt(2)
        for (auto& val : silhouette) {
            val *= std::sqrt(2.0);
        }

        return silhouette;
    }
};

#endif // PERSISTENCE_SILHOUETTE_H
