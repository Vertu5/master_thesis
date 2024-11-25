#ifndef ARGOS_SIMULATION_FORCOMPARATION_H
#define ARGOS_SIMULATION_FORCOMPARATION_H

#include <string>
#include <vector>
#include <map>
#include <pugixml.hpp>

class ArgosFile {
private:
    std::string name;
    pugi::xml_document doc;
    pugi::xml_node root;

public:
    ArgosFile(const std::string& name);
    void setDuration(const std::string& duration);
    void setSeed(const std::string& seed);
    void setLog(const std::string& log);
    void setFsm(const std::string& fsm);
    void setGenome(const std::string& genome);
    void setSim();
    void setPr();
    void run();
};

std::vector<std::string> extractFSM(const std::string& cs_dir);
std::string to_lowercase(const std::string& str);
std::pair<std::string, double> runSimulations(const std::string& mission, const std::string& method, const std::string& environment,
                   const std::vector<std::string>& seeds, const std::string& cs_dir, const std::string& argos_file_name);

#endif // ARGOS_SIMULATION_FORCOMPARATION_H