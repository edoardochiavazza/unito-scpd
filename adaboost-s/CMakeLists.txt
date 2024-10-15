cmake_minimum_required(VERSION 3.27)

# +=============================================+
# +============+ Setup Basic Infos +============+
# +=============================================+
set(CMAKE_CXX_STANDARD 17)
set(PROJECT_NAME AdaBoost)
project(${PROJECT_NAME})
set(EXECUTABLE_NAME ${PROJECT_NAME})

# Change default build/lib directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj")
file(GLOB HEADERS src/*.hpp)
file(GLOB SOURCES src/*.cpp)

include_directories(/opt/homebrew/include/)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})
