 // needed only for using library routines
 //
#include <mpi.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <cereal/archives/binary.hpp>  // Include Cereal

// TODO: Gather tree and formatting code, extract serialize and deserialize of tree in a method
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
    int rows = local_dataset.n_rows;
    int cols = local_dataset.n_cols;

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

// Funzioni per inviare e ricevere alberi
void send_tree(const mlpack::DecisionTree<>& tree, const int& dest_rank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::ostringstream oss;
    {
        cereal::BinaryOutputArchive oarchive(oss);
        oarchive(tree);
    }
    std::string serialized_data = oss.str();
    size_t length = serialized_data.size();

    MPI_Send(&length, 1, MPI_UNSIGNED_LONG, dest_rank, 0, MPI_COMM_WORLD);
    MPI_Send(serialized_data.data(), length, MPI_BYTE, dest_rank, 1, MPI_COMM_WORLD);
    std::cout << "RANK " << rank <<" try to send tree to "<< dest_rank << std::endl;
}


mlpack::DecisionTree<> receive_tree(const int& rank)
{

    size_t length;
    MPI_Recv(&length, 1, MPI_UNSIGNED_LONG, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<char> buffer(length);

    MPI_Recv(buffer.data(), length, MPI_BYTE, rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mlpack::DecisionTree<> tree; {
        std::istringstream iss(std::string(buffer.begin(), buffer.end()));
        cereal::BinaryInputArchive iarchive(iss);
        iarchive(tree);
    }
    std::cout << " MASTER  tree ricevuto "<<std::endl;
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

        // Serializza l'albero in un flusso in memoria
        std::stringstream ss;
        {
            cereal::BinaryOutputArchive archive(ss);
            tree.serialize(archive, 0);
        }

        // Estrai i dati serializzati come stringa binaria
        std::string serializedData = ss.str();
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
    MPI_Bcast(buffer.data(), dataSize, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Deserializzazione del modello in tutti i processi
    mlpack::DecisionTree<> b_tree;
    {
        std::stringstream ss(std::string(buffer.begin(), buffer.end()));
        cereal::BinaryInputArchive archive(ss);
        b_tree.serialize(archive, 0);
    }
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
    auto unique_w = arma::unique(new_result);
    std::cout << unique_w<< std::endl;

    arma::rowvec new_weights =  new_result % weights;
    weights = (new_weights / arma::sum(new_weights));
}

std::vector<double> gather_tree(const mlpack::DecisionTree<>& tree, const int rank, const int world_size, int num_elements) {
    int rank, worldSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &worldSize);

    // Serializzare l'albero locale in una stringa
    std::ostringstream oss;
    {
        cereal::BinaryOutputArchive oarchive(oss);
        localTree.serialize(oarchive);
    }
    std::string serializedTree = oss.str();

    // Determinare la lunghezza del buffer serializzato
    size_t localLength = serializedTree.size();
    std::vector<size_t> lengths(worldSize, 0);

    // Raccolta delle lunghezze nel processo root
    MPI_Gather(&localLength, 1, MPI_UNSIGNED_LONG, lengths.data(), 1, MPI_UNSIGNED_LONG, rootRank, comm);

    // Raccolta dei dati serializzati
    std::vector<char> localBuffer(serializedTree.begin(), serializedTree.end());
    std::vector<char> recvBuffer;
    std::vector<int> displacements(worldSize, 0);

    if (rank == rootRank) {
        // Calcolo delle dimensioni totali e dei displacements
        size_t totalSize = 0;
        for (int i = 0; i < worldSize; ++i) {
            displacements[i] = totalSize;
            totalSize += lengths[i];
        }
        recvBuffer.resize(totalSize);
    }

    // Gather dei dati serializzati
    MPI_Gatherv(localBuffer.data(), localLength, MPI_CHAR,
                recvBuffer.data(), lengths.data(), displacements.data(), MPI_CHAR, rootRank, comm);

    // Deserializzare gli alberi raccolti nel processo root
    std::vector<TreeType> gatheredTrees;
    if (rank == rootRank) {
        for (int i = 0; i < worldSize; ++i) {
            std::string serializedTree(&recvBuffer[displacements[i]], lengths[i]);
            std::istringstream iss(serializedTree);
            cereal::BinaryInputArchive iarchive(iss);

            TreeType tree;
            tree.serialize(iarchive);
            gatheredTrees.push_back(std::move(tree));
        }
    }

    return gatheredTrees;
}

std::vector<double> gather_trees_error(const std::vector<double>& local_data, const int rank, const int world_size, int num_elements) {

    std::vector<double> recvBuffer;
    if (rank == 0) {
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

 int main(int argc, char ** argv)
 {
     int rank, world_size;
     bool idd_settings_enviroment = true;
     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      std::vector<double> local_data;

        // master
        if(rank == 0){

            arma::mat train_dataset;
            arma::Row<size_t> train_labels;
            mlpack::data::DatasetInfo info;
            //Distribution data
            std::cout << "Loading training data..." << std::endl;
            load_datasets_and_labels(train_dataset, train_labels, info);
            int n_example = train_dataset.n_cols;
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
            std::cout<<"MASTER: In attesa di alberi "<<std::endl;
            std::vector<mlpack::DecisionTree<>> trees_m;
            for (int i = 1; i < world_size; ++i) {
                mlpack::DecisionTree<> tree = receive_tree(i);
                trees_m.push_back(tree);
            }

            for (int i = 0; i < world_size - 1; ++i) {
                std::cout<<"brodcast tree from master n: "<<i<<std::endl;
                broadcast_tree(trees_m[i]);
            }
                std::vector<double> trees_error = gather_trees_error(trees_error,rank, world_size,0);

        } else{ // client
            arma::mat client_training_dataset = receive_data_from_master();
            arma::Row<size_t> client_labels = arma::conv_to<arma::Row<size_t>>::from(client_training_dataset.row(client_training_dataset.n_rows - 1));
            client_training_dataset.shed_row(client_training_dataset.n_rows - 1);
            /*
            std::cout<< "RANK " << rank << " clientlabel n "<<client_labels.size()<<std::endl;
            std::cout<< "RANK " << rank << " datapoints n "<<client_training_dataset.n_cols<<std::endl;
            std::cout<< "RANK " << rank << " features n "<<client_training_dataset.n_rows<<std::endl;
            */
            arma::rowvec weights(client_training_dataset.n_cols, arma::fill::ones);
            weights /= client_training_dataset.n_cols;
            arma::Row<size_t> unique_labels = arma::unique(client_labels);
            mlpack::DecisionTree tree;
            tree.Train(client_training_dataset,client_labels, unique_labels.size(), weights, 10, 1e-7, 10);
            //std::cout << "Train tree client n: " << rank <<std::endl;
            send_tree(tree,0);
            std::vector<mlpack::DecisionTree<>> trees;
            //std::cout<< "RANK " << rank << " number trees receveid " << trees.size() <<std::endl;
            for(int i = 0; i < world_size - 1; ++i) {
                tree = broadcast_tree(tree);
                trees.push_back(tree);
                std::cout<< "Broadcast tree n: " << i <<std::endl;
            }
            std::cout<< "RANK " << rank << " number trees receveid " << trees.size() <<std::endl;
            arma::Row<size_t> predictions;
            std::vector<double> trees_error;
            for(auto tree : trees) {
                tree.Classify(client_training_dataset, predictions);
                arma::rowvec train_result = arma::conv_to<arma::rowvec>::from(predictions == client_labels);
                double total_error = calculate_total_error(train_result, weights);
                //std::cout<< "RANK " << rank << " total_error = "<< total_error << std::endl;
                trees_error.push_back(total_error);
            }
            gather_trees_error(trees_error,rank,world_size, trees_error.size());
        }
        std::cout<< "RANK " << rank << " finito "<<std::endl;
        MPI_Finalize();
     return 0;
 }





