#include <mpi.h>
#include <mlpack/core.hpp>
#include <iostream>
#include "communication_lib/libcomm.hpp"
#include "data_lib/datalib.hpp"


int main(int argc, char ** argv) {
    int rank, world_size, n_class;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::pair<mlpack::DecisionTree<>,double>> ensemble_learning;
    arma::mat client_training_dataset;
    arma::Row<size_t> client_labels, unique_labels;
    arma::rowvec weights;
    mlpack::data::DatasetInfo info;
    constexpr int epoch = 5;
    arma::mat temp_train;

    if (rank == 0) {
        arma::mat client_dataset;
        // Master process
        arma::mat train_dataset;
        arma::Row<size_t> train_labels;

        std::cout << "Loading training data..." << std::endl;
        load_datasets_and_labels(train_dataset, train_labels, info);
        unique_labels = arma::unique(train_labels);
        n_class = static_cast<int>(unique_labels.n_elem);

        int n_example = static_cast<int>(train_dataset.n_cols);

        int perc_n_example = (n_example / world_size);

        for (int i = 0; i < world_size; ++i) {
            train_dataset = shuffle(train_dataset, 1); // Shuffle columns
            client_dataset = train_dataset.cols(0, perc_n_example);
            if( i != 0) {
                send_data_to_client(client_dataset, i);
            }else {
                temp_train = client_dataset;
            }
        }
        client_dataset = temp_train;
        broadcastDatasetInfo(info);
    } else {
        // Client processes
        client_training_dataset = receive_data_from_master();
        info = broadcastDatasetInfo(info);
    }
    if (rank == 0) {
        client_training_dataset = temp_train;
    }
    client_labels = arma::conv_to<arma::Row<size_t>>::from(client_training_dataset.row(client_training_dataset.n_rows - 1));
    client_training_dataset.shed_row(client_training_dataset.n_rows - 1);
    arma::rowvec w_temp(client_training_dataset.n_cols, arma::fill::ones);
    weights = w_temp / static_cast<double>(client_training_dataset.n_cols);
    unique_labels = arma::unique(client_labels);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int e = 0; e < epoch; ++e) {
        mlpack::DecisionTree<> tree;
        tree.Train(client_training_dataset, info, client_labels, unique_labels.size(), weights, 10, 1e-7, 10);
        std::vector<mlpack::DecisionTree<>> vector_received_trees = broadcast_t(tree);

        //Calcolate total error
        std::vector<double> vector_total_errors;
        for(const mlpack::DecisionTree<>& t : vector_received_trees) {
            arma::Row<size_t> predictions;
            t.Classify(client_training_dataset, predictions);
            arma::rowvec train_result = arma::conv_to<arma::rowvec>::from(predictions == client_labels);
            double total_error = calculate_total_error(train_result, weights);
            vector_total_errors.push_back(total_error);
        }
        arma::mat m = broadcast_vec_total_error(vector_total_errors);
        int best_model_index = index_best_model(m);
        std::vector<int> best_trees_index = broadcast_index_best_tree(best_model_index);
        best_model_index = get_tree_from_majority_vote(best_trees_index);
        mlpack::DecisionTree<> best_tree_epoch = vector_received_trees[best_model_index];
        arma::Row<size_t> predictions;
        best_tree_epoch.Classify(client_training_dataset, predictions);
        arma::rowvec train_result = arma::conv_to<arma::rowvec>::from(predictions == client_labels);
        double total_error = calculate_total_error(train_result, weights);
        std::vector<double> total_clients_errors = broadcast_total_error_best_tree(total_error);
        double mean_total_error = std::reduce(total_clients_errors.begin(), total_clients_errors.end()) / world_size;
        double alpha = calculate_alpha(mean_total_error,static_cast<int>(unique_labels.size()),rank);
        ensemble_learning.emplace_back(best_tree_epoch,alpha);
        calculate_new_weights(train_result, alpha, weights);
    }
    if (rank == 0) {
        std::cout << "MASTER: Calculates the accuracy of the ensamble and its trees "<<std::endl;
        std::cout <<"Ensabmle learning size = "<< ensemble_learning.size() <<std::endl;

        arma::mat testDataset;
        arma::Row<size_t> test_labels;

        load_testData_and_labels(testDataset,test_labels,info);

        accuracy_for_model(ensemble_learning, testDataset, test_labels);
        arma::Mat<size_t> en_result = predict_all_dataset(ensemble_learning, testDataset);
        double accuracy_ensabmle = accuracy_ensamble(en_result, ensemble_learning, test_labels);
        std::cout << "Accuracy Ensabmle = " << accuracy_ensabmle <<" after " << epoch <<" epochs" <<std::endl;
    }
    MPI_Finalize();
}
