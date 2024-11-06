cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(TDAAnalysis)

# Set the output directory for the executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/..)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Find required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)
find_package(Boost REQUIRED)
find_package(CGAL REQUIRED)
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)

# Define Hera include directories
set(HERA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/hera/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/hera/include/hera
    ${CMAKE_CURRENT_SOURCE_DIR}/hera/wasserstein/include
    ${CMAKE_CURRENT_SOURCE_DIR}/hera/bottleneck/include
    ${CMAKE_CURRENT_SOURCE_DIR}/hera/extern
)

# Hera configuration
add_library(hera INTERFACE)
target_include_directories(hera INTERFACE ${HERA_INCLUDE_DIRS})

# Add the naturalHHD subdirectory
add_subdirectory(naturalHHD/cpp-v1.0)


# Include directories for the main project
include_directories(
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/naturalHHD/cpp-v1.0/src
    ${HERA_INCLUDE_DIRS}
    ${CGAL_INCLUDE_DIRS}
    ${GMP_INCLUDE_DIRS}
    ${MPFR_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/data_loader.cpp
    src/ChampVectoriel.cpp
    #src/topology_analysis.cpp
)

# Add executable
add_executable(analyze_topology ${SOURCES})

# Set properties
set_target_properties(analyze_topology PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Link libraries
target_link_libraries(analyze_topology
    PRIVATE
    Eigen3::Eigen
    hera
    Boost::boost
    naturalHHD
    ${CGAL_LIBRARIES}
    gmp
    mpfr
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# Add compile options
if(OpenMP_CXX_FOUND)
    target_compile_options(analyze_topology PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Add debug symbols in Release mode
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(analyze_topology PRIVATE -g)
endif()
