 // needed only for using library routines
 //
#include <mpi.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <cereal/archives/binary.hpp>  // Include Cereal


// TODO: send info matrix
int random_federate_distribution(){

    //generate a number between 70 and 100, that will be the percentage of dataset own by the client
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> random_distribution_number(70, 101);
    int random_distr = random_distribution_number(gen);
    return  random_distr;

}

void load_datasets_and_labels(arma::mat &train_dataset, arma::Row<size_t>& train_labels, mlpack::data::DatasetInfo& info) {

    const std::string train_path = "/home/edoardo/Desktop/unito-scpd/adaboost-mpi/datasets/covertype.train.arff";
    const std::string train_labels_path = "/home/edoardo/Desktop/unito-scpd/adaboost-mpi/datasets/covertype.train.labels.csv";
    mlpack::data::Load(train_path, train_dataset, info, true);
    mlpack::data::Load(train_labels_path, train_labels, true);
    arma::rowvec labels_vec = arma::conv_to<arma::rowvec>::from(arma::conv_to<arma::Row<double>>::from(train_labels));
    train_dataset.insert_rows(train_dataset.n_rows, labels_vec);
}

void send_data_to_client(const arma::mat &local_dataset,const int &client_rank) {
    const int rows = static_cast<int>(local_dataset.n_rows);
    const int cols = static_cast<int>(local_dataset.n_cols);

    //std::cout << "Sending matrix to "<< client_rank <<std::endl;
    // Invia le dimensioni della matrice
    MPI_Send(&rows, 1, MPI_INT, client_rank, 0, MPI_COMM_WORLD);
    MPI_Send(&cols, 1, MPI_INT, client_rank, 0, MPI_COMM_WORLD);

    // Invia i dati della matrice come un array di double
    MPI_Send(local_dataset.memptr(), rows * cols, MPI_DOUBLE, client_rank, 0, MPI_COMM_WORLD);

}

arma::mat receive_data_from_master() {
    int rank;
    int rows;
    int cols;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Recv(&rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(&cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    arma::mat received_dataset(rows, cols);
    MPI_Recv(received_dataset.memptr(), rows * cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //std::cout << "Received data client n "<< rank <<std::endl;
    return received_dataset;
}

template <typename T>
std::string serialize_obj(T &obj) {
    std::ostringstream oss;
    {
        cereal::BinaryOutputArchive oarchive(oss);
        oarchive(obj);
    }
    std::string serialized_data = oss.str();
    return serialized_data;
}


mlpack::DecisionTree<> deserialize_tree (const std::string& serializedTree) {
    mlpack::DecisionTree<> tree; {
        std::istringstream iss(std::string(serializedTree.begin(), serializedTree.end()));
        cereal::BinaryInputArchive iarchive(iss);
        iarchive(tree);
    }
    return tree;
}

// Funzioni per inviare e ricevere alberi
void send_tree(mlpack::DecisionTree<>& tree, const int& dest_rank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    std::string serialized_data = serialize_obj(tree);
    size_t length = serialized_data.size();

    MPI_Send(&length, 1, MPI_UNSIGNED_LONG, dest_rank, 0, MPI_COMM_WORLD);
    MPI_Send(serialized_data.data(), static_cast<int>(length), MPI_BYTE, dest_rank, 1, MPI_COMM_WORLD);
    std::cout << "RANK " << rank <<" try to send tree to "<< dest_rank << std::endl;
}


mlpack::DecisionTree<> receive_tree(const int& rank)
{

    size_t length;
    MPI_Recv(&length, 1, MPI_UNSIGNED_LONG, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<char> buffer(length);

    MPI_Recv(buffer.data(), static_cast<int>(length), MPI_BYTE, rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mlpack::DecisionTree<> tree = deserialize_tree(std::string(buffer.begin(), buffer.end()));
    std::cout << " MASTER  tree ricevuto "<<std::endl;
    buffer.clear();
    return tree;
}


 mlpack::DecisionTree<> broadcast_tree(mlpack::DecisionTree<> &tree, MPI_Comm comm = MPI_COMM_WORLD)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Buffer che conterrà i dati serializzati
    std::vector<char> buffer;
    size_t dataSize;

    if (rank == 0)  // Processo root
    {

        std::string serializedData = serialize_obj(tree);
        dataSize = serializedData.size();

        // Copia i dati in un buffer per MPI_Bcast
        buffer.assign(serializedData.begin(), serializedData.end());
    }

    // Broadcast della dimensione dei dati
    MPI_Bcast(&dataSize, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Alloca memoria per il buffer sui processi non-root
    if (rank != 0)
    {
        buffer.resize(dataSize);
    }

    // Broadcast dei dati binari
    MPI_Bcast(buffer.data(), static_cast<int>(dataSize), MPI_CHAR, 0, MPI_COMM_WORLD);

    // Deserializzazione del modello in tutti i processi
    mlpack::DecisionTree<> b_tree = deserialize_tree(std::string(buffer.begin(), buffer.end()));
    return b_tree;
}

double calculate_total_error(const arma::rowvec& train_result, const arma::rowvec& weights){
    double total_error = arma::sum((1.0 - train_result) % weights);
    //std::cout << total_error<< std::endl;
    double epsilon = 1e-10;
    if (total_error == 0) {
        total_error = epsilon;
    } else if (total_error == 1) {
        total_error = 1 - epsilon;
    }
    return total_error;

}

double calculate_alpha(const double& total_error , const int& n_class){
    double alpha = log((1.0 - total_error) / total_error) + log(n_class);
    if (std::isnan(alpha) || std::isinf(alpha)) {
        std::cerr << "Errore: alpha non valido" << std::endl;
        return -1 ;
    }
    return alpha;

}

void calculate_new_weights(const arma::rowvec& train_result, const double& alpha, arma::rowvec& weights){
    arma::rowvec new_result = arma::exp(alpha * (1.0 - train_result));
    //new_result.print();
    auto unique_w = arma::unique(new_result);
    //unique_w.print();
    arma::rowvec new_weights =  new_result % weights;
    weights = new_weights / arma::sum(new_weights);
}


 std::vector<mlpack::DecisionTree<>> gather_tree(mlpack::DecisionTree<> &tree, const int rank, const int& world_size) {

    size_t localLength;
    std::vector<char> localBuffer;

    if (rank != 0) {
        std::string serializedTree;
        serializedTree = serialize_obj(tree);
        // Determinare la lunghezza del buffer serializzato
        localLength = serializedTree.size();
        localBuffer.resize(localLength);
        // Copia i dati della stringa nel buffer
        std::copy(serializedTree.begin(), serializedTree.end(), localBuffer.begin());
    }else {
        localLength = 0;
    }

    std::vector<size_t> lengths(world_size, 0);

    // Raccolta delle lunghezze nel processo root
    MPI_Gather(&localLength, 1, MPI_UNSIGNED_LONG, lengths.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    std::vector<char> recvBuffer;
    std::vector<int> recvLengths(world_size, 0);
    std::vector<int> displacements(world_size, 0);

    if (rank == 0) {
        // Calcolo delle dimensioni totali e dei displacements
        int totalSize = 0;
        for (int i = 0; i < world_size; ++i) {
            displacements[i] = totalSize;
            totalSize += static_cast<int>(lengths[i]);
            recvLengths[i] = static_cast<int>(lengths[i]);
        }
        recvBuffer.resize(totalSize);
    }

    // Gather dei dati serializzati
    MPI_Gatherv(localBuffer.data(), static_cast<int>(localLength), MPI_CHAR,
                recvBuffer.data(), recvLengths.data(), displacements.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

    // Deserializzare gli alberi raccolti nel processo root
    std::vector<mlpack::DecisionTree<>> gatheredTrees;
    if (rank == 0) {
        for (int i = 1; i < world_size; ++i) {
            std::string serializedTree(&recvBuffer[displacements[i]], lengths[i]);
            mlpack::DecisionTree<> tree_recev = deserialize_tree(serializedTree);
            gatheredTrees.push_back(tree_recev);
        }
    }
    return gatheredTrees;
}

std::vector<double> gather_trees_error(const std::vector<double>& local_data, const int& rank, const int& world_size, const int& num_elements) {

    std::vector<double> recvBuffer;
    if (rank == 0) {
        recvBuffer.clear();
        // Il processo 0 dovrà ricevere N elementi da ogni altro processo
        recvBuffer.resize( (world_size - 1) * (world_size - 1 ));  // N per (size - 1) perché il processo 0 non invia nulla
    }

    // MPI_Gatherv per raccogliere i dati da tutti i processi
    std::vector<int> sendCounts(world_size, world_size - 1);  // Ogni processo invia un vettore di N elementi
    std::vector<int> displacements(world_size, 0);

    if (rank == 0) {
        sendCounts[0] = 0;
        // Per il processo 0, i dati saranno disposti uno dopo l'altro
        for (int i = 1; i < world_size; ++i) {
            displacements[i] = displacements[i - 1] + sendCounts[i - 1];
        }

    }

    MPI_Gatherv(local_data.data(), num_elements, MPI_DOUBLE, recvBuffer.data(), sendCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return recvBuffer;
}

int index_best_model(const arma::mat & trees_error){
    trees_error.print();
    arma::rowvec col_sums = arma::sum(trees_error, 0); // Somma lungo le righe (direzione verticale)

    arma::uword max_col_index = col_sums.index_max();
    std::cout << max_col_index << std::endl;
    return static_cast<int>(max_col_index);
}

void broadcast_alpha(double& alpha) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Broadcasting alpha
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


// Funzione per deserializzare l'oggetto DatasetInfo
mlpack::data::DatasetInfo deserialize_dataset_info(const std::string& serialized_info) {
    mlpack::data::DatasetInfo info; {
        std::istringstream iss(std::string(serialized_info.begin(), serialized_info.end()));
        cereal::BinaryInputArchive iarchive(iss);
        iarchive(info);
    }
    return info;
}

mlpack::data::DatasetInfo broadcastDatasetInfo(mlpack::data::DatasetInfo& info) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   std::string serializedData;

    if (rank == 0) {
        // Serializza il DatasetInfo
        serializedData = serialize_obj(info);
    }

    // Broadcast della dimensione del buffer
    size_t dataSize = serializedData.size();
    MPI_Bcast(&dataSize, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Ridimensiona il buffer per i processi non-root
    if (rank != 0) {
        serializedData.resize(dataSize);
    }

    // Broadcast del buffer serializzato
    MPI_Bcast(serializedData.data(), static_cast<int>(dataSize), MPI_CHAR, 0, MPI_COMM_WORLD);

    // Deserializza nei processi non-root
    if (rank != 0) {
        info = deserialize_dataset_info(serializedData);
    }
    return info;
}

 int main(int argc, char ** argv)
 {
     int rank, world_size,n_class;
     bool idd_settings_enviroment = true;
     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     std::vector<double> local_data;
     std::vector<mlpack::DecisionTree<>> ensable_learning;
     arma::mat client_training_dataset;
     arma::Row<size_t> client_labels, unique_labels;
     arma::rowvec weights;
     mlpack::data::DatasetInfo info;
     double alpha;
     constexpr int epoch = 5;
        // master
        if(rank == 0) {
            arma::mat train_dataset;
            arma::Row<size_t> train_labels;
            //Distribution data
            std::cout << "Loading training data..." << std::endl;
            load_datasets_and_labels(train_dataset, train_labels, info);
            unique_labels = arma::unique(train_labels);
            n_class = static_cast<int>(unique_labels.n_elem);
            int n_example = static_cast<int>(train_dataset.n_cols);
            int perc_n_example;
            if(idd_settings_enviroment) {
                perc_n_example = n_example / (world_size - 1) / 100;
            }
            for (int i = 1; i < world_size; ++i) {
                if(!idd_settings_enviroment) {
                    //settings non-idd
                    int random_per_datataset = random_federate_distribution();
                    perc_n_example = n_example * random_per_datataset / 100;
                }
                arma::mat shuffled_train_dataset = shuffle(train_dataset, 1); // Shuffle sulle colonne (dimensione 1)
                arma::mat client_dataset = shuffled_train_dataset.cols(0, perc_n_example);
                send_data_to_client(client_dataset, i);
                //end distribution
            }
            broadcastDatasetInfo(info);
        }else {
            client_training_dataset = receive_data_from_master();
            info = broadcastDatasetInfo(info);
            //std::cout<< "RANK " << rank << " INFO n "<<info.Dimensionality()<<std::endl;
            client_labels = arma::conv_to<arma::Row<size_t>>::from(client_training_dataset.row(client_training_dataset.n_rows - 1));
            client_training_dataset.shed_row(client_training_dataset.n_rows - 1);
            /*
            std::cout<< "RANK " << rank << " clientlabel n "<<client_labels.size()<<std::endl;
            std::cout<< "RANK " << rank << " datapoints n "<<client_training_dataset.n_cols<<std::endl;
            std::cout<< "RANK " << rank << " features n "<<client_training_dataset.n_rows<<std::endl;
            */
            arma::rowvec w_temp(client_training_dataset.n_cols, arma::fill::ones);
            weights = w_temp;
            weights /= static_cast<double>(client_training_dataset.n_cols);
            unique_labels = arma::unique(client_labels);
            //std::cout<< "RANK " << rank << " labels n "<<unique_labels<<std::endl;
        }
    MPI_Barrier(MPI_COMM_WORLD);
    for(int t = 0; t < epoch; ++t) {
        if(rank == 0) {
            std::cout<<"MASTER: In attesa di alberi "<<std::endl;
            mlpack:: DecisionTree<> model;
            std::vector<mlpack::DecisionTree<>> trees_m = gather_tree(model, rank, world_size);
            for (int i = 0; i < world_size - 1; ++i) {
                //std::cout<<"brodcast tree from master n: "<<i<<std::endl;
                broadcast_tree(trees_m[i]);
            }
            //std::cout<< "RANK " << rank << " gather_trees_error "<<std::endl;
            std::vector<double> client_tree_errors = gather_trees_error(local_data, rank, world_size,0);

            // Conversione in arma::vec e creazione matrice 2x4
            arma::vec arma_vec(client_tree_errors);
            arma::mat arma_mat = arma::reshape(arma_vec, world_size - 1, world_size - 1).t(); // 2 righe, 4 colonne
            int best_model_index = index_best_model(arma_mat);
            ensable_learning.push_back(trees_m[best_model_index]);
            double average_total_errors = arma::mean(arma_mat.col(best_model_index));
            alpha = calculate_alpha(average_total_errors, n_class);
            broadcast_alpha(alpha);
            std::cout<< "RANK " << rank << " brodcast_alpha value: "<< alpha<<std::endl;
            broadcast_tree(ensable_learning[t]);

            //std::cout<< "RANK " << rank << " brodcast_alpha_tree"<<std::endl;
        }else{ // client
            mlpack::DecisionTree tree;
            tree.Train(client_training_dataset,info,client_labels, unique_labels.size(), weights, 10, 1e-7, 10);
            //std::cout << "Train tree client n: " << rank <<std::endl;
            gather_tree(tree, rank, world_size);

            std::vector<mlpack::DecisionTree<>> trees;
            //std::cout<< "RANK " << rank << " number trees receveid " << trees.size() <<std::endl;
            for(int i = 0; i < world_size - 1; ++i) {
                tree = broadcast_tree(tree);
                trees.push_back(tree);

            }
            //std::cout<< "RANK " << rank << " number trees receveid " << trees.size() <<std::endl;
            arma::Row<size_t> predictions;
            std::vector<double> trees_error;
            for(const auto& t_i : trees) {
                t_i.Classify(client_training_dataset, predictions);
                arma::rowvec train_result = arma::conv_to<arma::rowvec>::from(predictions == client_labels);
                double total_error = calculate_total_error(train_result, weights);
                //std::cout<< "RANK " << rank << " total_error = "<< total_error << std::endl;
                trees_error.push_back(total_error);
            }/*
            for (auto r : trees_error) {
                std::cout<< "RANK " << rank << " total_error = " << r<< std::endl;
            }*/
            gather_trees_error(trees_error,rank,world_size, static_cast<int>(trees_error.size()));

            //std::cout<< "RANK " << rank << " gather_trees_error "<<std::endl;
            broadcast_alpha(alpha);
            mlpack::DecisionTree<> best_tree = broadcast_tree(best_tree);
            //std::cout<< "RANK " << rank << " broadcast_tree "<<std::endl;
            ensable_learning.push_back(best_tree);
            best_tree.Classify(client_training_dataset, predictions);
            //std::cout<< "RANK " << rank << " tree_calssify "<<std::endl;
            arma::rowvec train_result = arma::conv_to<arma::rowvec>::from(predictions == client_labels);
            arma::rowvec old_weights = weights;
            calculate_new_weights(train_result, alpha,weights);
            old_weights = old_weights - weights;
            /*
            for(auto d : old_weights) {
                std::cout<< "RANK " << rank << " difference_weights = " << d<< std::endl;
            }
            */
        }
        if(rank == 0) {
            std::cout<< "Epoch " << t << " end "<<std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);

    }
    if(rank == 0) {
        const std::string test_path = "/home/edoardo/Desktop/unito-scpd/adaboost-mpi/datasets/covertype.test.arff";
        const std::string test_labels_path = "/home/edoardo/Desktop/unito-scpd/adaboost-mpi/datasets/covertype.test.labels.csv";
        arma::mat testDataset;
        arma::Row<size_t> test_labels, test_predictions;
        mlpack::data::Load(test_path, testDataset, info, true);
        mlpack::data::Load(test_labels_path, test_labels, true);

        for(const auto & i : ensable_learning) {
            i.Classify(testDataset,test_predictions);
            auto prediction_result = arma::conv_to<arma::rowvec>::from( test_predictions == test_labels);
            double true_predictions = arma::sum(prediction_result == 1.0) * 1.0;
            float accuracy = true_predictions / prediction_result.n_elem;
            std::cout<<"Accuracy = " << accuracy << std::endl;
        }
    }
    MPI_Finalize();
     return 0;
 }





