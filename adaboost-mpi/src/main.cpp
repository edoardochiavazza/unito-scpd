#include <mpi.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <iostream>
#include <vector>
#include "communication_lib/libcomm.hpp"
#include "data_lib/datalib.hpp"

int main(int argc, char** argv) {
    int rank, world_size, n_class;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<std::pair<mlpack::DecisionTree<>,double>> ensemble_learning;
    arma::mat client_training_dataset;
    arma::Row<size_t> client_labels, unique_labels;
    arma::rowvec weights;
    mlpack::data::DatasetInfo info;
    arma::mat train_dataset;
    arma::Row<size_t> train_labels;
    double alpha;
    int epochs[7] = {5,10,20,30,40,50,100};
    double average_total_error;

    if (rank == 0) {
        // Master process


        std::cout << "MASTER: Uploads and distributes data among "<< world_size - 1 <<" clients " << std::endl;
        load_datasets_and_labels(train_dataset, train_labels, info);

        unique_labels = arma::unique(train_labels);
        n_class = static_cast<int>(unique_labels.n_elem);

	//arma::mat inflatedData = train_dataset;
        //for (size_t i = 1; i < weak_scale_factor * world_size; ++i) {
            //inflatedData = arma::join_rows(inflatedData, train_dataset);
        //}
        	
        int n_example = static_cast<int>(train_dataset.n_cols);
        int perc_n_example = n_example / (world_size - 1);

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
    for(auto e: epochs){
        auto start = std::chrono::high_resolution_clock::now();
        double average_time_epoch = 0;
        for (int t = 0; t < e; ++t) {
            double time_epoch = 0;
            auto start_epoch = std::chrono::high_resolution_clock::now();
            if (rank == 0) {
                // Master process
                std::vector<double> local_data;
                mlpack::DecisionTree<> model;
                std::vector<mlpack::DecisionTree<>> trees_m = gather_tree(model, rank, world_size);

                for (int i = 0; i < world_size - 1; ++i) {
                    broadcast_tree(trees_m[i]);
                }

                std::vector<double> client_tree_errors = gather_trees_error(local_data, rank, world_size, 0);

                arma::vec arma_vec(client_tree_errors);
                arma::mat arma_mat = arma::reshape(arma_vec, world_size - 1, world_size - 1).t();

                int best_model_index = index_best_model(arma_mat);

                average_total_error = arma::mean(arma_mat.col(best_model_index));
                alpha = calculate_alpha(average_total_error, n_class);
		        if(alpha > 0.01){
			        ensemble_learning.emplace_back(trees_m[best_model_index],alpha);
			        std::cout << "MASTER:added tree "<<std::endl;
		        }
                broadcast_alpha(alpha);
                broadcast_tree(trees_m[best_model_index]);

            } else {
                // Client processes
                mlpack::DecisionTree<> tree;
                tree.Train(client_training_dataset, info, client_labels, unique_labels.size(), weights, 20, 1e-3, 0);

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
		        if(alpha > 0.01){
			        ensemble_learning.emplace_back(best_tree,alpha);
		        }
                best_tree.Classify(client_training_dataset, predictions);
                arma::rowvec train_result = arma::conv_to<arma::rowvec>::from(predictions == client_labels);
                calculate_new_weights(train_result, alpha, weights);
            }
            if (rank == 0) {
                std::cout << "MASTER: Epoch " << t << " end" <<" with average total error = " << average_total_error << " and alpha = " << alpha<<std::endl;
            }
            auto end_epoch_timer = std::chrono::high_resolution_clock::now();
            time_epoch =  std::chrono::duration<double>(end_epoch_timer - start_epoch).count();
            average_time_epoch = (average_time_epoch + time_epoch)/2;
        }
        auto end_total_timer = std::chrono::high_resolution_clock::now();
        double time_total = std::chrono::duration<double>(end_total_timer - start).count();
        if (rank == 0) {
            std::cout << "MASTER: Calculates the accuracy of the ensamble and its trees "<<std::endl;
            std::cout <<"Ensabmle learning size = "<< ensemble_learning.size() <<std::endl;
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
            // Nome del file di output
            std::string fileName = "../res/risultati_adaboost-mpi_strong_13_prest.txt";
	    int num_node = std::atoi(argv[1]);
            int num_task_for_node = std::atoi(argv[2]);	    
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
        ensemble_learning.clear();
        arma::rowvec w_temp(client_training_dataset.n_cols, arma::fill::ones);
        weights = w_temp / static_cast<double>(client_training_dataset.n_cols);
    }


    MPI_Finalize();
    return 0;

