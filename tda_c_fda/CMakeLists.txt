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
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Armadillo REQUIRED)

# Define Hera include directories
set(HERA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/hera/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/hera/include/hera
    ${CMAKE_CURRENT_SOURCE_DIR}/hera/wasserstein/include
    ${CMAKE_CURRENT_SOURCE_DIR}/hera/bottleneck/include
    ${CMAKE_CURRENT_SOURCE_DIR}/hera/extern
)

# FDASRSF sources
set(FDASRSF_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/fdasrsf_python/src/dp_grid.c
    ${CMAKE_CURRENT_SOURCE_DIR}/fdasrsf_python/src/dp_nbhd.c
    ${CMAKE_CURRENT_SOURCE_DIR}/fdasrsf_python/src/misc_funcs.c
    ${CMAKE_CURRENT_SOURCE_DIR}/fdasrsf_python/src/DynamicProgrammingQ2.c
)

# Create FDASRSF library
add_library(fdasrsf STATIC ${FDASRSF_SOURCES})
target_include_directories(fdasrsf PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/fdasrsf_python/src
)
set_target_properties(fdasrsf PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Hera configuration
add_library(hera INTERFACE)
target_include_directories(hera INTERFACE ${HERA_INCLUDE_DIRS})

# Add the naturalHHD subdirectory
add_subdirectory(naturalHHD/cpp-v1.0)

# Include directories
include_directories(
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/naturalHHD/cpp-v1.0/src
    ${CMAKE_CURRENT_SOURCE_DIR}/fdasrsf_python/src
    ${HERA_INCLUDE_DIRS}
    ${CGAL_INCLUDE_DIRS}
    ${GMP_INCLUDE_DIRS}
    ${MPFR_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIRS}
)

# Common source files
set(COMMON_SOURCES
    src/data_loader.cpp
)

# Define executables sources
set(EXPERT_SOURCES
    src/main.cpp
    ${COMMON_SOURCES}
)

set(IMITATION_SOURCES
    src/analyze_topology.cpp
    ${COMMON_SOURCES}
)

set(EXTENDED_PERSISTENCE
    src/main_extended.cpp
    ${COMMON_SOURCES}
)

set(PERSISTENCE_ANALYSIS_SOURCES
    src/persistence_analyser.cpp
    src/SimulationProcessor.cpp
    ${COMMON_SOURCES}
)

# Add executables
add_executable(expert_behavior ${EXPERT_SOURCES})
add_executable(analyze_topology ${IMITATION_SOURCES})
add_executable(extendedcsv ${EXTENDED_PERSISTENCE})
add_executable(persistence_analysis ${PERSISTENCE_ANALYSIS_SOURCES})

# Set properties for all executables
foreach(TARGET expert_behavior analyze_topology extendedcsv persistence_analysis)
    set_target_properties(${TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    
    # Link libraries
    target_link_libraries(${TARGET}
        PRIVATE
        Eigen3::Eigen
        hera
        Boost::boost
        naturalHHD
        ${CGAL_LIBRARIES}
        gmp
        mpfr
        fdasrsf
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
    )
    
    # OpenMP options
    if(OpenMP_CXX_FOUND)
        target_compile_options(${TARGET} PRIVATE ${OpenMP_CXX_FLAGS})
    endif()
    
    # Debug symbols in Release mode
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${TARGET} PRIVATE -g)
    endif()
endforeach()
