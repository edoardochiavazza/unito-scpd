cmake_minimum_required(VERSION 3.27)
set(CMAKE_CXX_STANDARD 17)


set(PROJECT_NAME adaboost_mpi)
project(${PROJECT_NAME})
set(EXECUTABLE_NAME ${PROJECT_NAME})
# Change default build/lib directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj")

find_package(MPI REQUIRED)

file(GLOB HEADERS src/*.hpp)
file(GLOB SOURCES src/*.cpp)


add_library(communication_lib STATIC)
target_sources(communication_lib PRIVATE
        src/communication_lib/libcomm.cpp
        src/communication_lib/libcomm.hpp)
target_include_directories(communication_lib PUBLIC src/communication_lib)


add_library(data_lib STATIC)
target_sources(data_lib PRIVATE
        src/data_lib/datalib.cpp
        src/data_lib/datalib.hpp
)
target_include_directories(data_lib PUBLIC src/data_lib)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})
target_link_libraries(adaboost_mpi PUBLIC MPI::MPI_CXX)

