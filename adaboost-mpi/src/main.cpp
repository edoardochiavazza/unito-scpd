#include <mpi.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include "communication_lib/libcomm.hpp"
#include "data_lib/datalib.hpp"

int main(int argc, char** argv) {
    int rank, world_size, n_class;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<mlpack::DecisionTree<>> ensemble_learning;
    arma::mat client_training_dataset;
    arma::Row<size_t> client_labels, unique_labels;
    arma::rowvec weights;
    mlpack::data::DatasetInfo info;
    double alpha;
    constexpr int epoch = 5;

    if (rank == 0) {
        // Master process
        arma::mat train_dataset;
        arma::Row<size_t> train_labels;

        std::cout << "Loading training data..." << std::endl;
        load_datasets_and_labels(train_dataset, train_labels, info);
        unique_labels = arma::unique(train_labels);
        n_class = static_cast<int>(unique_labels.n_elem);

        int n_example = static_cast<int>(train_dataset.n_cols);
        int perc_n_example = n_example / (world_size - 1) / 100;

        for (int i = 1; i < world_size; ++i) {
            arma::mat shuffled_train_dataset = shuffle(train_dataset, 1); // Shuffle columns
            arma::mat client_dataset = shuffled_train_dataset.cols(0, perc_n_example);
            send_data_to_client(client_dataset, i);
        }

        broadcastDatasetInfo(info);
    } else {
        // Client processes
        client_training_dataset = receive_data_from_master();
        info = broadcastDatasetInfo(info);

        client_labels = arma::conv_to<arma::Row<size_t>>::from(client_training_dataset.row(client_training_dataset.n_rows - 1));
        client_training_dataset.shed_row(client_training_dataset.n_rows - 1);

        arma::rowvec w_temp(client_training_dataset.n_cols, arma::fill::ones);
        weights = w_temp / static_cast<double>(client_training_dataset.n_cols);
        unique_labels = arma::unique(client_labels);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int t = 0; t < epoch; ++t) {
        if (rank == 0) {
            // Master process
            std::vector<double> local_data;
            std::cout << "MASTER: Waiting for trees..." << std::endl;

            mlpack::DecisionTree<> model;
            std::vector<mlpack::DecisionTree<>> trees_m = gather_tree(model, rank, world_size);

            for (int i = 0; i < world_size - 1; ++i) {
                broadcast_tree(trees_m[i]);
            }

            std::vector<double> client_tree_errors = gather_trees_error(local_data, rank, world_size, 0);

            arma::vec arma_vec(client_tree_errors);
            arma::mat arma_mat = arma::reshape(arma_vec, world_size - 1, world_size - 1).t();

            int best_model_index = index_best_model(arma_mat);
            ensemble_learning.push_back(trees_m[best_model_index]);

            double average_total_errors = arma::mean(arma_mat.col(best_model_index));
            alpha = calculate_alpha(average_total_errors, n_class);

            broadcast_alpha(alpha);
            std::cout << "MASTER: Broadcasted alpha value: " << alpha << std::endl;
            broadcast_tree(ensemble_learning[t]);

        } else {
            // Client processes
            mlpack::DecisionTree<> tree;
            tree.Train(client_training_dataset, info, client_labels, unique_labels.size(), weights, 10, 1e-7, 10);

            gather_tree(tree, rank, world_size);

            std::vector<mlpack::DecisionTree<>> trees;
            for (int i = 0; i < world_size - 1; ++i) {
                tree = broadcast_tree(tree);
                trees.push_back(tree);
            }

            arma::Row<size_t> predictions;
            std::vector<double> trees_error;

            for (const auto& t_i : trees) {
                t_i.Classify(client_training_dataset, predictions);
                arma::rowvec train_result = arma::conv_to<arma::rowvec>::from(predictions == client_labels);
                double total_error = calculate_total_error(train_result, weights);
                trees_error.push_back(total_error);
            }

            gather_trees_error(trees_error, rank, world_size, static_cast<int>(trees_error.size()));
            broadcast_alpha(alpha);

            mlpack::DecisionTree<> best_tree = broadcast_tree(best_tree);
            ensemble_learning.push_back(best_tree);

            best_tree.Classify(client_training_dataset, predictions);
            arma::rowvec train_result = arma::conv_to<arma::rowvec>::from(predictions == client_labels);
            calculate_new_weights(train_result, alpha, weights);
        }

        if (rank == 0) {
            std::cout << "Epoch " << t << " end" << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const std::string test_path = "/home/edoardo/Desktop/unito-scpd/adaboost-mpi/datasets/covertype.test.arff";
        const std::string test_labels_path = "/home/edoardo/Desktop/unito-scpd/adaboost-mpi/datasets/covertype.test.labels.csv";

        arma::mat testDataset;
        arma::Row<size_t> test_labels, test_predictions;

        mlpack::data::Load(test_path, testDataset, info, true);
        mlpack::data::Load(test_labels_path, test_labels, true);

        for (const auto& i : ensemble_learning) {
            i.Classify(testDataset, test_predictions);
            auto prediction_result = arma::conv_to<arma::rowvec>::from(test_predictions == test_labels);
            double true_predictions = static_cast<double>(arma::sum(prediction_result == 1.0));
            double accuracy = true_predictions / static_cast<double>(prediction_result.n_elem);
            std::cout << "Accuracy = " << accuracy << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
