cmake_minimum_required(VERSION 3.10)
project(ArgosRunner)

# Set the output directory for the executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/..)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ARGoS-related settings
set(ARGOS_ROOT /home/ndinga/master/argos3-dist)
set(ARGOS_INCLUDE_DIR ${ARGOS_ROOT}/include)
set(ARGOS_LIBRARY_DIR ${ARGOS_ROOT}/lib/argos3)

include_directories(${ARGOS_INCLUDE_DIR})
include_directories(/usr/local/include)
include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pugixml/src)
link_directories(${ARGOS_LIBRARY_DIR})

# Source files
set(SOURCES
    src/main.cpp
    src/ArgosSimulation.cpp
    pugixml/src/pugixml.cpp
)

# Add executable with new name
add_executable(simulate_missions ${SOURCES})

# Set the output directory for the executable
set_target_properties(simulate_missions PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
)

# Find required ARGoS libraries
find_library(ARGOS_CORE_LIBRARY NAMES argos3core_simulator PATHS ${ARGOS_LIBRARY_DIR} NO_DEFAULT_PATH REQUIRED)

# Link libraries
target_link_libraries(simulate_missions
    ${ARGOS_CORE_LIBRARY}
)

# Set rpath
set_target_properties(simulate_missions PROPERTIES
    INSTALL_RPATH "${ARGOS_LIBRARY_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Find and link OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(simulate_missions OpenMP::OpenMP_CXX)
endif()
