#include <mpi.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <iostream>
#include <vector>
#include "communication_lib/libcomm.hpp"
#include "data_lib/datalib.hpp"
#include <algorithm>
#include <limits>

int main(int argc, char ** argv) {
    int rank, world_size, n_class;
    int epochs[7] = {5,10,20,30,40,50,100};
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::pair<mlpack::DecisionTree<>,double>> ensemble_learning;
    arma::mat client_training_dataset;
    arma::Row<size_t> client_labels, unique_labels;
    arma::rowvec weights;
    mlpack::data::DatasetInfo info;
 
    if (rank == 0) {
        // Master process
        arma::mat train_dataset;
        arma::Row<size_t> train_labels;
	    int num_node = std::atoi(argv[1]);
        std::cout << "Loading training data..." << std::endl;
        load_datasets_and_labels(train_dataset, train_labels, info);
        int n_example = static_cast<int>(train_dataset.n_cols);
        int perc_n_example = (n_example / world_size);
	std::cout << "inflated data start" << std::endl;
	arma::mat data_replicated = arma::repmat(train_dataset, 1, num_node);
	train_dataset = data_replicated;
	std::cout << "inflated data end" << std::endl;
	    std::cout << "send start"<< std::endl;
        for (int i = 1; i < world_size; ++i) {
            train_dataset = shuffle(train_dataset, 1); // Shuffle columns
            client_training_dataset = train_dataset.cols(0, perc_n_example);
            send_data_to_client(client_training_dataset, i);
        }
        broadcastDatasetInfo(info);
        train_dataset = shuffle(train_dataset, 1); // Shuffle columns
        client_training_dataset = train_dataset.cols(0, perc_n_example);
    } else {
        // Client processes
        client_training_dataset = receive_data_from_master();
        info = broadcastDatasetInfo(info);
    }

    client_labels = arma::conv_to<arma::Row<size_t>>::from(client_training_dataset.row(client_training_dataset.n_rows - 1));
    client_training_dataset.shed_row(client_training_dataset.n_rows - 1);
    unique_labels = arma::unique(client_labels);
    n_class = static_cast<int>(unique_labels.n_elem);
    arma::rowvec w_temp(client_training_dataset.n_cols, arma::fill::ones);
    weights = w_temp / static_cast<double>(client_training_dataset.n_cols);
    MPI_Barrier(MPI_COMM_WORLD);
    for (auto e : epochs){
        auto start = std::chrono::high_resolution_clock::now();
        double average_time_epoch = 100000;
        for (int i = 0; i < e; ++i) {
            double time_epoch = 0;
            auto start_epoch = std::chrono::high_resolution_clock::now();
            mlpack::DecisionTree<> tree;
            tree.Train(client_training_dataset, info, client_labels, unique_labels.size(), weights, 20, 1e-3, 0);
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
            double mean_total_error = average_total_error_best_tree(total_error);
            double alpha = calculate_alpha(mean_total_error,static_cast<int>(unique_labels.size()));
            calculate_new_weights(train_result, alpha, weights);
	    ensemble_learning.emplace_back(best_tree_epoch, alpha);
            auto end_epoch_timer = std::chrono::high_resolution_clock::now();
            time_epoch =  std::chrono::duration<double>(end_epoch_timer - start_epoch).count();
            average_time_epoch = std::min(average_time_epoch, time_epoch);
        }
        auto end_total_timer = std::chrono::high_resolution_clock::now();
        double time_total = std::chrono::duration<double>(end_total_timer - start).count();
        if (rank == 0) {
            std::cout << "MASTER: Calculates the accuracy of the ensamble and its trees "<<std::endl;
            std::cout <<"Ensabmle learning size = "<< ensemble_learning.size() <<std::endl;
            arma::mat train_dataset;
            arma::Row<size_t> train_labels;
            arma::mat testDataset;
            arma::Row<size_t> test_labels;
            mlpack::data::DatasetInfo info_test, info_train;
            load_datasets_and_labels(train_dataset,train_labels, info_test);
            load_testData_and_labels(testDataset,test_labels,info_train);
            accuracy_for_model(ensemble_learning, testDataset, test_labels);
            arma::Mat<size_t> en_result = predict_all_dataset(ensemble_learning, testDataset);
            double accuracy_ensabmle_test = accuracy_ensamble(en_result, ensemble_learning, test_labels, n_class);
            en_result = predict_all_dataset(ensemble_learning, train_dataset);
            double accuracy_ensabmle_train = accuracy_ensamble(en_result, ensemble_learning, train_labels, n_class);
            std::cout << "Accuracy Ensabmle = " << accuracy_ensabmle_test <<" for the test dataset "<< " in " << e <<" epochs"<<std::endl;
            std::cout << "Accuracy Ensabmle = " << accuracy_ensabmle_train <<" for the train dataset "<< " in " << e << " epochs"<<std::endl;
	        int num_node = std::atoi(argv[1]);
            int num_task_for_node = std::atoi(argv[2]);
            // Nome del file di output
            std::string fileName = "../res/risultati_adaboost-mpi-v2_w9.txt";

            // Creazione di un oggetto di tipo ofstream
            std::ofstream outputFile(fileName, std::ios::app);

            // Controllo che il file sia stato aperto correttamente
            if (!outputFile.is_open()) {
                std::cerr << "Errore nell'apertura del file: " << fileName << std::endl;
                return 1;
            }

            // Scrittura dei risultati nel file

            outputFile << "--------------------------\n";
            outputFile << "Machine: Broadwell\n";
            outputFile << "Num nodes: "<< num_node<<"\n";
            outputFile << "Num tasks per node: "<<num_task_for_node<<"\n";
            outputFile << "Total tasks:"<<num_node * num_task_for_node<<"\n";
            outputFile << "Number epoch: "<< e<<"\n";
            outputFile << "Time epoch (T1): " << average_time_epoch << " seconds\n";
            outputFile << "Time epochs (T1): " << time_total << " seconds\n";
            outputFile << "Ensamble accuracy = " << accuracy_ensabmle_test <<" for test dataset\n";
            outputFile << "Ensamble accuracy = " << accuracy_ensabmle_train <<" for training dataset\n";
            // Chiusura del file
            outputFile.close();
            std::cout << "Risultati scritti con successo nel file: " << fileName <<" per il valore " << e<<std::endl;
        }
        weights = w_temp / static_cast<double>(client_training_dataset.n_cols);
        ensemble_learning.clear();
    }
    MPI_Finalize();
}
