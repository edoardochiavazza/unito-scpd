 // needed only for using library routines
 //
#include <mpi.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <cereal/archives/binary.hpp>  // Include Cereal

//  TODO : implementato la serializzazione dell'albero. Passo sucessivo i client mandano l'ablero all'aggregatore che li BRODCASTA a tutti i client(magari escludere quello che ha già(salvarli in base al rank??)). Poi li testano su il proprio dt
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

    std::cout << "Sending matrix to "<< client_rank <<std::endl;
    // Invia le dimensioni della matrice
    MPI_Send(&rows, 1, MPI_INT, client_rank, 0, MPI_COMM_WORLD);
    MPI_Send(&cols, 1, MPI_INT, client_rank, 0, MPI_COMM_WORLD);

    // Invia i dati della matrice come un array di double
    MPI_Send(local_dataset.memptr(), rows * cols, MPI_DOUBLE, client_rank, 0, MPI_COMM_WORLD);

}

arma::mat receive_data_from_master() {
    int rows;
    int cols;
    MPI_Recv(&rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(&cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    arma::mat received_dataset(rows, cols);
    MPI_Recv(received_dataset.memptr(), rows * cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Funzioni per inviare e ricevere alberi
void send_tree(const mlpack::DecisionTree<>& tree, int dest_rank)
{
    std::ostringstream oss;
    {
        cereal::BinaryOutputArchive oarchive(oss);
        oarchive(tree);
    }
    std::string serialized_data = oss.str();
    size_t length = serialized_data.size();

    MPI_Send(&length, 1, MPI_UNSIGNED_LONG, dest_rank, 0, MPI_COMM_WORLD);
    MPI_Send(serialized_data.data(), length, MPI_BYTE, dest_rank, 1, MPI_COMM_WORLD);
}
std::istringstream receive_serialized_tree()
{
    size_t length;
    MPI_Recv(&length, 1, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Status status;
    std::vector<char> buffer(length);
    MPI_Recv(buffer.data(), length, MPI_BYTE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::istringstream iss(std::string(buffer.begin(), buffer.end()));
    return iss;
}

mlpack::DecisionTree<> receive_tree()
{

    int rank;
    MPI_Recv(&rank, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    size_t length;
    MPI_Recv(&length, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<char> buffer(length);
    MPI_Recv(buffer.data(), length, MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    mlpack::DecisionTree<> tree;
    std::istringstream iss(std::string(buffer.begin(), buffer.end()));
    {
        cereal::BinaryInputArchive iarchive(iss);
        iarchive(tree);
    }

    return tree;
}

// Funzione per inviare un intero vettore di alberi
mlpack::DecisionTree<>  brodcast_tree(const std::istringstream& tree)
{
    int rank;
    std::string serialized_tree = tree.str();
    size_t length = serialized_tree.size();
    MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(serialized_tree.data(), length, MPI_BYTE, 0, MPI_COMM_WORLD);
    mlpack::DecisionTree<> tree;
    if(rank != 0) {
        std::istringstream iss(serialized_tree);
        {
            cereal::BinaryInputArchive iarchive(iss);
            iarchive(tree);
        }
    }
    return tree;

}

void send_tree_to_client(const std::vector<mlpack::DecisionTree<>>& trees, int dest_rank) {

}

 int main(int argc, char ** argv)
 {
     int rank, world_size;
     bool idd_settings_enviroment = true;
     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);


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

         for (int i = 1; i < world_size; ++i) {
             auto[tree, rank] = receive_serialized_tree();
             brodcast_tree(tree, rank);
         }

     } else{ // client

         mlpack::data::DatasetInfo info;
         arma::mat client_training_dataset = receive_data_from_master();
         arma::Row<size_t> client_labels = arma::conv_to<arma::Row<size_t>>::from(client_training_dataset.col(client_training_dataset.n_cols - 1).t());
         client_training_dataset.shed_col(client_training_dataset.n_cols - 1);
         arma::rowvec weights(client_training_dataset.n_cols, arma::fill::ones);
         weights /= client_training_dataset.n_cols; //2.4588e-06
         arma::Row<size_t> unique_labels = arma::unique(client_labels);
         arma::Row<size_t> predictions;

         std::cout << "Train tree from client n: " << rank <<std::endl;
         mlpack::DecisionTree tree(client_training_dataset, info, client_labels, unique_labels.size(), weights, 10, 1e-7, 10);
         tree.Classify(client_training_dataset, predictions);
         arma::rowvec train_result = arma::conv_to<arma::rowvec>::from(predictions == client_labels);
         send_tree(tree,0);

         int sender_client_rank;
         std::vector<mlpack::DecisionTree<>> trees;
         for(int i = 1; i < world_size; ++i) {
             brodcast_tree(tree, sender_client_rank);
         }




     }
     MPI_Finalize();

     return 0;
 }





