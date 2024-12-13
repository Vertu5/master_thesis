#ifndef PERSISTENCE_SILHOUETTE_H
#define PERSISTENCE_SILHOUETTE_H

#include <vector>
#include <functional>
#include <limits>
#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>

// Forward declarations
class DiagramScaler;

using PersistenceDiagram = std::vector<std::pair<double, double>>;

class PersistenceSilhouette {
public:
    using WeightFunction = std::function<double(const std::pair<double, double>&)>;

    struct Range {
        std::vector<double> range;
        
        Range(const std::vector<double>& r = {std::numeric_limits<double>::quiet_NaN(), 
                                            std::numeric_limits<double>::quiet_NaN()}) 
            : range(r) {}
        
        bool has_nan() const {
            return std::isnan(range[0]) || std::isnan(range[1]);
        }
    };

    PersistenceSilhouette(
        WeightFunction weight = [](const std::pair<double, double>&) { return 1.0; },
        size_t resolution = 100,
        Range sample_range = Range(),
        bool keep_endpoints = false
    ) : weight_(weight), 
        resolution_(resolution), 
        sample_range_(sample_range),
        keep_endpoints_(keep_endpoints),
        is_fitted_(false) {}

    // Fit method, similar to Python's fit
    void fit(const std::vector<PersistenceDiagram>& diagrams) {
        if (sample_range_.has_nan()) {
            grid_from_sample_range(diagrams);
        }
        is_fitted_ = true;
    }

    // Transform method, similar to Python's transform
    std::vector<std::vector<double>> transform(
        const std::vector<PersistenceDiagram>& diagrams) {
        if (!is_fitted_) {
            fit(diagrams);
        }

        std::vector<std::vector<double>> result;
        result.reserve(diagrams.size());
        
        for (const auto& diagram : diagrams) {
            result.push_back(compute_silhouette(diagram));
        }
        return result;
    }

    // Equivalent to Python's __call__
    std::vector<double> operator()(const PersistenceDiagram& diagram) {
        if (!is_fitted_) {
            std::vector<PersistenceDiagram> diagrams{diagram};
            fit(diagrams);
        }
        return compute_silhouette(diagram);
    }

    // Accessors
    const std::vector<double>& grid() const { return grid_; }
    Range get_sample_range() const { return sample_range_; }
    size_t resolution() const { return resolution_; }
    bool is_fitted() const { return is_fitted_; }

private:
    WeightFunction weight_;
    size_t resolution_;
    Range sample_range_;
    bool keep_endpoints_;
    bool is_fitted_;
    std::vector<double> grid_;
    size_t new_resolution_;

    void grid_from_sample_range(const std::vector<PersistenceDiagram>& diagrams) {
        auto sample_range = sample_range_.range;
        new_resolution_ = resolution_;
        
        if (!keep_endpoints_) {
            new_resolution_ += (sample_range_.has_nan() ? 2 : 0);
        }

        auto fixed_range = automatic_sample_range(sample_range, diagrams);
        
        if (fixed_range[0] != fixed_range[1]) {
            grid_ = linspace(fixed_range[0], fixed_range[1], new_resolution_);
        } else {
            grid_ = std::vector<double>(new_resolution_, fixed_range[0]);
        }

        if (!keep_endpoints_) {
            trim_endpoints();
        }
    }

    std::vector<double> automatic_sample_range(
        const std::vector<double>& sample_range,
        const std::vector<PersistenceDiagram>& diagrams) {
        
        if (!sample_range_.has_nan()) {
            return sample_range;
        }

        if (diagrams.empty() || diagrams[0].empty()) {
            double b = std::max({sample_range[0], sample_range[1], 
                               -std::numeric_limits<double>::infinity()});
            return {b, b};
        }

        // Implement MinMax scaling similar to Python version
        double min_x = std::numeric_limits<double>::max();
        double max_x = -std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double max_y = -std::numeric_limits<double>::max();

        for (const auto& diagram : diagrams) {
            for (const auto& point : diagram) {
                min_x = std::min(min_x, point.first);
                max_x = std::max(max_x, point.first);
                min_y = std::min(min_y, point.second);
                max_y = std::max(max_y, point.second);
            }
        }

        return {min_x, max_y};
    }

    void trim_endpoints() {
        if (grid_.size() > 2) {
            double step = (grid_.back() - grid_.front()) / (grid_.size() - 1);
            grid_.front() += step;
            grid_.back() -= step;
        }
    }

    std::vector<double> linspace(double start, double stop, size_t num) {
        std::vector<double> result(num);
        double step = (stop - start) / (num - 1);
        
        for (size_t i = 0; i < num; i++) {
            result[i] = start + i * step;
        }
        return result;
    }

    std::vector<double> compute_silhouette(const PersistenceDiagram& diagram) {
        if (diagram.empty()) {
            return std::vector<double>(resolution_, 0.0);
        }

        // Calculate midpoints and heights
        std::vector<double> midpoints;
        std::vector<double> heights;
        std::vector<double> weights;
        
        midpoints.reserve(diagram.size());
        heights.reserve(diagram.size());
        weights.reserve(diagram.size());

        double total_weight = 0.0;
        
        for (const auto& point : diagram) {
            midpoints.push_back((point.first + point.second) / 2.0);
            heights.push_back((point.second - point.first) / 2.0);
            double w = weight_(point);
            weights.push_back(w);
            total_weight += w;
        }

        // Compute silhouette values
        std::vector<double> silhouette(grid_.size(), 0.0);
        
        for (size_t i = 0; i < grid_.size(); ++i) {
            double x = grid_[i];
            for (size_t j = 0; j < diagram.size(); ++j) {
                silhouette[i] += (weights[j] / total_weight) * 
                                std::max(heights[j] - std::abs(x - midpoints[j]), 0.0);
            }
            silhouette[i] *= std::sqrt(2.0);
        }

        return silhouette;
    }
};

#endif // PERSISTENCE_SILHOUETTE_H