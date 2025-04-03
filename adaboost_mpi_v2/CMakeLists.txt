cmake_minimum_required(VERSION 3.27)
set(CMAKE_CXX_STANDARD 17)

set(PROJECT_NAME adaboost_mpi_v2)
project(${PROJECT_NAME})
set(EXECUTABLE_NAME ${PROJECT_NAME})

# Change default build/lib directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj")

# Find MPI
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(PkgConfig REQUIRED)
pkg_check_modules(MLPACK REQUIRED mlpack)
include_directories(${MLPACK_INCLUDE_DIRS})
link_directories(${MLPACK_LIBRARY_DIRS})
add_definitions(${MLPACK_CFLAGS_OTHER})


# Sources
file(GLOB HEADERS src/*.hpp)
file(GLOB SOURCES src/*.cpp)

# Communication library
add_library(communication_lib STATIC)
target_sources(communication_lib PUBLIC
        src/communication_lib/libcomm.cpp
        src/communication_lib/libcomm.hpp)
target_include_directories(communication_lib PUBLIC src/communication_lib)
target_link_libraries(communication_lib PUBLIC MPI::MPI_CXX)

# Data library
add_library(data_lib STATIC)
target_sources(data_lib PUBLIC
        src/data_lib/datalib.cpp
        src/data_lib/datalib.hpp)
target_include_directories(data_lib PUBLIC src/data_lib)
target_link_libraries(data_lib PUBLIC MPI::MPI_CXX)

# Main executable
add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX data_lib communication_lib ${mlpack_LIBRARIES})
