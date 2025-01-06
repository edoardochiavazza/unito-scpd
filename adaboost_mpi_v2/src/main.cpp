#include <mpi.h>
#include <mlpack/core.hpp>
#include <iostream>
#include "communication_lib/libcomm.hpp"
#include "data_lib/datalib.hpp"


int main(int argc, char ** argv)
{
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
        int perc_n_example = n_example / (world_size) / 100;
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
    std::cout<<"Rank " << rank <<" unique: " << unique_labels.n_elem << std::endl;
    int num_class;
    MPI_Barrier(MPI_COMM_WORLD);
    arma::Row<size_t> num_classes;
    for (int i = 0; i < world_size; ++i) {
        num_class = static_cast<int>(unique_labels.n_elem);
        auto t = broadcast_num_class(num_class, rank);
        num_classes.insert_cols(num_classes.n_cols,t);
    }
    size_t num_class_c = arma::max(num_classes);
    std::cout<<"Rank " << rank <<" numclass: " <<num_class_c << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    mlpack::DecisionTree<> tree;
    tree.Train(client_training_dataset, info, client_labels, num_class_c, weights, 10, 1e-7, 10);
    std::vector<mlpack::DecisionTree<>> vector_received_trees;
    for (int i = 0; i < world_size; ++i) {
        auto c = broadcast_tree(tree, i);
        vector_received_trees.push_back(tree);
    }
    std::cout<<"Rank " << rank <<" size: " <<vector_received_trees.size() << std::endl;
    MPI_Finalize();
}
