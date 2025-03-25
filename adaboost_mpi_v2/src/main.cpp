#include <mpi.h>
#include <mlpack/core.hpp>
#include <iostream>
#include "communication_lib/libcomm.hpp"
#include "data_lib/datalib.hpp"


int main(int argc, char ** argv) {
    int rank, world_size;
    int epochs[9] = {5,10,20,30,40,50,100,500,1000};
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::pair<mlpack::DecisionTree<>,double>> ensemble_learning;
    arma::mat client_training_dataset;
    arma::Row<size_t> client_labels, unique_labels;
    arma::rowvec weights;
    mlpack::data::DatasetInfo info;
    arma::mat temp_train;

    if (rank == 0) {
        // Master process
        arma::mat train_dataset;
        arma::Row<size_t> train_labels;

        std::cout << "Loading training data..." << std::endl;
        load_datasets_and_labels(train_dataset, train_labels, info);

        train_dataset = repmat(train_dataset, world_size, 1);

        int n_example = static_cast<int>(train_dataset.n_cols);
        int perc_n_example = (n_example / world_size);

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
    arma::rowvec w_temp(client_training_dataset.n_cols, arma::fill::ones);
    weights = w_temp / static_cast<double>(client_training_dataset.n_cols);
    unique_labels = arma::unique(client_labels);
    MPI_Barrier(MPI_COMM_WORLD);
    for (auto e : epochs){
        auto start = std::chrono::high_resolution_clock::now();
        double average_time_epoch = 0;
        for (int i = 0; i < e; ++i) {
            double time_epoch = 0;
            auto start_epoch = std::chrono::high_resolution_clock::now();
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
            if(rank ==0) {
                m.print();
            }
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
            double alpha = calculate_alpha(mean_total_error,static_cast<int>(unique_labels.size()));
            ensemble_learning.emplace_back(best_tree_epoch,alpha);
            calculate_new_weights(train_result, alpha, weights);
            auto end_epoch_timer = std::chrono::high_resolution_clock::now();
            time_epoch =  std::chrono::duration<double>(end_epoch_timer - start_epoch).count();
            average_time_epoch = (average_time_epoch + time_epoch)/2;
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
            double accuracy_ensabmle_test = accuracy_ensamble(en_result, ensemble_learning, test_labels);
            en_result = predict_all_dataset(ensemble_learning, train_dataset);
            double accuracy_ensabmle_train = accuracy_ensamble(en_result, ensemble_learning, train_labels);
            std::cout << "Accuracy Ensabmle = " << accuracy_ensabmle_test <<" for the test dataset "<< " in " << e <<" epochs"<<std::endl;
            std::cout << "Accuracy Ensabmle = " << accuracy_ensabmle_train <<" for the train dataset "<< " in " << e << " epochs"<<std::endl;
            // Nome del file di output
            std::string fileName = "../res/risultati_adaboost-mpi-v2.txt";

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
            outputFile << "Num nodes: 1 \n";
            outputFile << "Num tasks per node: 18 \n";
            outputFile << "Total tasks: 18 \n";
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
