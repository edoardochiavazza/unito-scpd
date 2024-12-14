//
// Created by edoardo on 12/12/24.
//

#pragma once

#include <armadillo>
#include <mlpack/core.hpp>

// Function to load the training dataset and labels
void load_datasets_and_labels(arma::mat &train_dataset, arma::Row<size_t>& train_labels, mlpack::data::DatasetInfo& info);

// Function to calculate the total error based on the training results and weights
double calculate_total_error(const arma::rowvec& train_result, const arma::rowvec& weights);

// Function to calculate the alpha parameter based on the total error and number of classes
double calculate_alpha(const double& total_error , const int& n_class);

// Function to calculate the new weights based on the training results and alpha
void calculate_new_weights(const arma::rowvec& train_result, const double& alpha, arma::rowvec& weights);

// Function to find the index of the best model (the one with the minimum error)
int index_best_model(const arma::mat & trees_error);