// include/topology_analysis.h
#ifndef PERSISTENCE_CALCULATOR_H
#define PERSISTENCE_CALCULATOR_H

#pragma once
#include "topology_analysis.h"
#include <gudhi/Persistent_cohomology.h>

struct PersistenceResult {
    double distance;
    std::map<int, std::vector<std::pair<double, double>>> diagrams;
};

class PersistenceCalculator {
public:
    static PersistenceResult compareSurfaces(const Surface3D& surface1,
                                           const Surface3D& surface2);
                                           
    static void analyzePairs(const std::vector<std::pair<int, int>>& pairs,
                            const std::vector<MissionData>& mission_data);
};
#endif // PERSISTENCE_CALCULATOR_H