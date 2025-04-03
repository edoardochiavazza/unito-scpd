#pragma once


#include <mpi.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <cereal/archives/binary.hpp>
#include <armadillo>
#include <vector>
#include <string>
#include <sstream>

// Function to send dataset to the client in MPI communication
void send_data_to_client(const arma::mat &local_dataset, const int &client_rank);

// Function to receive dataset from the master node in MPI communication
arma::mat receive_data_from_master();

// Template function to serialize an object to a string
template <typename T>
std::string serialize_obj(T &obj);

// Function to deserialize a decision tree from a serialized string
mlpack::DecisionTree<> deserialize_tree(const std::string& serializedTree);

// Function to send a decision tree to a destination node in MPI communication
void send_tree(mlpack::DecisionTree<>& tree, const int& dest_rank);

// Function to receive a decision tree from a source node in MPI communication
mlpack::DecisionTree<> receive_tree(const int& rank);

// Function to broadcast a decision tree to all processes in MPI communication
mlpack::DecisionTree<> broadcast_tree(mlpack::DecisionTree<> &tree, const int &root_rank);

// Function to gather decision trees from all nodes in MPI communication
std::vector<mlpack::DecisionTree<>> gather_tree(mlpack::DecisionTree<> &tree, const int& rank, const int& world_size);

// Function to gather error values from all nodes in MPI communication
std::vector<double> gather_trees_error(const std::vector<double>& local_data, const int& rank, const int& world_size, const int& num_elements);

// Function to broadcast a scalar value (alpha) to all processes in MPI communication
void broadcast_alpha(double& alpha);

// Function to deserialize DatasetInfo object from a serialized string
mlpack::data::DatasetInfo deserialize_dataset_info(const std::string& serialized_info);

// Function to broadcast DatasetInfo object to all processes in MPI communication
mlpack::data::DatasetInfo broadcastDatasetInfo(mlpack::data::DatasetInfo& info);

// Function to broadcast num_class to all processes in MPI communication
int broadcast_num_class(int& num_class, const int& root_rank);

arma::mat broadcast_vec_total_error(const std::vector<double> &total_errors);

std::vector<mlpack::DecisionTree<>> broadcast_t(mlpack::DecisionTree<> t);

std::vector<int> broadcast_index_best_tree(const int& index);

double average_total_error_best_tree(const double& total_error);



