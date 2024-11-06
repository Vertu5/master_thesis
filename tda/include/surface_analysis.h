#pragma once

#include <vector>
#include <set>
#include <map>
#include <Eigen/Dense>
#include <gudhi/Alpha_complex.h>
#include <gudhi/Simplex_tree.h>
#include "VectorField.h"
#include "HHD.h"

// 3D point structure
struct Point3D {
    double x, y, z;
};

// Triangle structure to represent surface elements
struct Triangle {
    size_t v1, v2, v3;
    Triangle(size_t v1_, size_t v2_, size_t v3_) : v1(v1_), v2(v2_), v3(v3_) {}
};

// Structure to store surface data
struct MissionSurface {
    std::vector<Point3D> points;
    std::vector<Triangle> triangles;
    std::vector<double> values;
    std::map<std::string, std::vector<bool>> regions;

    MissionSurface() = default;
};

class SurfaceAnalysis {
public:
    static std::vector<std::vector<std::vector<double>>> points_to_grid(
        const std::vector<Point3D>& points,
        const std::vector<double>& values,
        size_t resolution = 50) 
    {
        if(points.empty()) {
            return std::vector<std::vector<std::vector<double>>>(1, 
                   std::vector<std::vector<double>>(1, 
                   std::vector<double>(1, 0.0)));
        }

        // Find bounds
        double x_min = points[0].x, x_max = points[0].x;
        double y_min = points[0].y, y_max = points[0].y;
        double z_min = points[0].z, z_max = points[0].z;

        for(const auto& point : points) {
            x_min = std::min(x_min, point.x);
            x_max = std::max(x_max, point.x);
            y_min = std::min(y_min, point.y);
            y_max = std::max(y_max, point.y);
            z_min = std::min(z_min, point.z);
            z_max = std::max(z_max, point.z);
        }

        // Small epsilon to avoid edge cases
        const double epsilon = 1e-10;
        x_max += epsilon; y_max += epsilon; z_max += epsilon;

        // Create and fill 3D grid
        std::vector<std::vector<std::vector<double>>> grid(
            resolution,
            std::vector<std::vector<double>>(
                resolution,
                std::vector<double>(resolution, 0.0)
            )
        );

        for(size_t i = 0; i < points.size(); ++i) {
            const auto& point = points[i];
            int x_idx = static_cast<int>((point.x - x_min)/(x_max - x_min) * (resolution-1));
            int y_idx = static_cast<int>((point.y - y_min)/(y_max - y_min) * (resolution-1));
            int z_idx = static_cast<int>((point.z - z_min)/(z_max - z_min) * (resolution-1));

            if(x_idx >= 0 && x_idx < static_cast<int>(resolution) &&
               y_idx >= 0 && y_idx < static_cast<int>(resolution) &&
               z_idx >= 0 && z_idx < static_cast<int>(resolution)) {
                grid[x_idx][y_idx][z_idx] = 1.0;  // Binary presence
            }
        }

        return grid;
    }

    static std::tuple<std::vector<Point3D>, 
                     std::vector<Triangle>, 
                     std::vector<double>, 
                     std::map<std::string, std::vector<bool>>>
    create_3d_surface(const VectorFieldData& field_data,
                     const naturalHHD<float>& hhd,
                     const std::string& component_name = "Divergent",
                     size_t min_surface_size = 5) 
    {
        // Get field dimensions and scale
        const size_t ny_bins = field_data.Ux.rows();
        const size_t nx_bins = field_data.Ux.cols();
        const double scale = 1.231;

        // Extract points and values from HHD result
        std::vector<Point3D> points;
        std::vector<double> values;
        
        for(size_t i = 0; i < ny_bins; ++i) {
            for(size_t j = 0; j < nx_bins; ++j) {
                if(field_data.mask(i,j)) {
                    size_t idx = i * nx_bins + j;
                    double x = 2 * scale * (static_cast<double>(j) / nx_bins - 0.5);
                    double y = -2 * scale * (static_cast<double>(i) / ny_bins - 0.5);
                    double z = (component_name == "Divergent") ? hhd.D[idx] : hhd.Ru[idx];
                    
                    points.push_back({x, y, z});
                    values.push_back(z);
                }
            }
        }

        // Compute standard deviation for region classification
        double std_dev = compute_standard_deviation(values);

        // Create points for Alpha complex (2D projection)
        using Kernel = CGAL::Epeck_d<CGAL::Dynamic_dimension_tag>;
        std::vector<typename Kernel::Point_d> cgal_points;
        for (const auto& point : points) {
            std::vector<double> coords = {point.x, point.y};
            cgal_points.push_back(typename Kernel::Point_d(coords.begin(), coords.end()));
        }

        // Create Alpha complex and extract triangulation
        Gudhi::alpha_complex::Alpha_complex<Kernel> alpha_complex(cgal_points);
        Gudhi::Simplex_tree<> stree;
        alpha_complex.create_complex(stree);

        // Classify points into regions
        size_t num_points = points.size();
        std::vector<bool> strong_positive(num_points);
        std::vector<bool> weak_positive(num_points);
        std::vector<bool> weak_negative(num_points);
        std::vector<bool> strong_negative(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            double val = values[i];
            if (val > std_dev) {
                strong_positive[i] = true;
            } else if (val > 0) {
                weak_positive[i] = true;
            } else if (val >= -std_dev) {
                weak_negative[i] = true;
            } else {
                strong_negative[i] = true;
            }
        }

        std::map<std::string, std::vector<bool>> regions = {
            {"strong_positive", strong_positive},
            {"weak_positive", weak_positive},
            {"weak_negative", weak_negative},
            {"strong_negative", strong_negative}
        };

        // Extract triangles by region
        std::vector<Triangle> all_triangles;
        for (auto sh : stree.skeleton_simplex_range(2)) {
            if (stree.dimension(sh) == 2) {
                std::vector<size_t> vertices;
                for (auto vertex : stree.simplex_vertex_range(sh)) {
                    vertices.push_back(vertex);
                }

                for (const auto& [region_name, region_mask] : regions) {
                    if (region_mask[vertices[0]] && region_mask[vertices[1]] && region_mask[vertices[2]]) {
                        all_triangles.push_back({vertices[0], vertices[1], vertices[2]});
                    }
                }
            }
        }

        // Filter connected surfaces
        auto filtered_triangles = identify_surface_triangles(all_triangles, min_surface_size);

        // Remove unused points and reindex
        std::set<size_t> used_indices;
        for (const auto& tri : filtered_triangles) {
            used_indices.insert(tri.v1);
            used_indices.insert(tri.v2);
            used_indices.insert(tri.v3);
        }

        std::map<size_t, size_t> index_mapping;
        std::vector<Point3D> new_points;
        std::vector<double> new_values;
        size_t new_idx = 0;

        for (size_t idx : used_indices) {
            index_mapping[idx] = new_idx++;
            new_points.push_back(points[idx]);
            new_values.push_back(values[idx]);
        }

        // Update triangles with new indices
        std::vector<Triangle> new_triangles;
        for (const auto& tri : filtered_triangles) {
            new_triangles.push_back({
                index_mapping[tri.v1],
                index_mapping[tri.v2],
                index_mapping[tri.v3]
            });
        }

        // Update regions
        std::map<std::string, std::vector<bool>> new_regions;
        for (const auto& [region_name, region_mask] : regions) {
            std::vector<bool> new_region_mask;
            for (size_t idx : used_indices) {
                new_region_mask.push_back(region_mask[idx]);
            }
            new_regions[region_name] = new_region_mask;
        }

        return {new_points, new_triangles, new_values, new_regions};
    }

private:
    static double compute_standard_deviation(const std::vector<double>& values) {
        if (values.empty()) return 0.0;

        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
        return std::sqrt(sq_sum / values.size() - mean * mean);
    }

    static std::vector<Triangle> identify_surface_triangles(
        const std::vector<Triangle>& triangles,
        size_t min_connected_triangles)
    {
        // Create triangle connectivity graph
        std::map<size_t, std::set<size_t>> triangle_neighbors;
        for (size_t i = 0; i < triangles.size(); ++i) {
            triangle_neighbors[i] = std::set<size_t>();
            const auto& tri1 = triangles[i];

            for (size_t j = 0; j < triangles.size(); ++j) {
                if (i == j) continue;
                const auto& tri2 = triangles[j];

                // Check for shared edge (2 shared vertices)
                std::set<size_t> vertices1 = {tri1.v1, tri1.v2, tri1.v3};
                std::set<size_t> vertices2 = {tri2.v1, tri2.v2, tri2.v3};
                std::set<size_t> shared_vertices;
                std::set_intersection(
                    vertices1.begin(), vertices1.end(),
                    vertices2.begin(), vertices2.end(),
                    std::inserter(shared_vertices, shared_vertices.begin())
                );

                if (shared_vertices.size() == 2) {
                    triangle_neighbors[i].insert(j);
                }
            }
        }

        // Find connected components using DFS
        std::vector<bool> visited(triangles.size(), false);
        std::vector<std::set<size_t>> components;

        auto dfs = [&](auto& self, size_t start, std::set<size_t>& component) -> void {
            visited[start] = true;
            component.insert(start);
            for (size_t neighbor : triangle_neighbors[start]) {
                if (!visited[neighbor]) {
                    self(self, neighbor, component);
                }
            }
        };

        for (size_t i = 0; i < triangles.size(); ++i) {
            if (!visited[i]) {
                std::set<size_t> component;
                dfs(dfs, i, component);
                if (component.size() >= min_connected_triangles) {
                    components.push_back(component);
                }
            }
        }

        // Collect triangles from valid components
        std::vector<Triangle> valid_triangles;
        for (const auto& component : components) {
            for (size_t idx : component) {
                valid_triangles.push_back(triangles[idx]);
            }
        }

        return valid_triangles;
    }
};