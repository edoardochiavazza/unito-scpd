add_library(libcomm libcomm.cpp)
target_include_directories(libcomm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(libcomm PUBLIC MPI::MPI_CXX Armadillo::Armadillo mlpack::mlpack)