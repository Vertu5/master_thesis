cmake_minimum_required(VERSION 3.10)
project(BinWidthOptimizer)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(/usr/local/include)
include_directories(include)

# Include filesystem if necessary
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lstdc++fs")
endif()

# Add OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add source files
set(SOURCES
    src/main.cpp
    src/data_loader.cpp
    src/vector_field.cpp
    src/bin_width_optimizer.cpp
)

# Add executable
add_executable(bin_width_optimizer ${SOURCES})

# Include directories
target_include_directories(bin_width_optimizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(bin_width_optimizer Eigen3::Eigen OpenMP::OpenMP_CXX)
