 // needed only for using library routines
 //
#include <mpi.h>
#include <mlpack.hpp>

int random_federate_distribution(){

    //generate a number between 70 and 100, that will be the percentage of dataset own by the client
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> random_distribution_number(70, 101);
    int random_distr = random_distribution_number(gen);
    return  random_distr;

}


 int main(int argc, char ** argv)
 {
     int rank, world_size;

     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);


     // master
     if(rank == 0){
         arma::mat train_dataset;
         arma::Row<size_t> train_labels;
         mlpack::data::DatasetInfo info;
         std::cout << "Loading training data..." << std::endl;
         const std::string train_path = "/home/edoardo/Desktop/unito-scpd/adaboost-mpi/datasets/covertype.train.arff";
         const std::string train_labels_path = "/home/edoardo/Desktop/unito-scpd/adaboost-mpi/datasets/covertype.train.labels.csv";
         Load(train_path, train_dataset, info, true);
         mlpack::data::Load(train_labels_path, train_labels, true);
         std::cout << train_dataset.n_rows << std::endl;
         arma::rowvec labels_vec = arma::conv_to<arma::rowvec>::from(arma::conv_to<arma::Row<double>>::from(train_labels));
         train_dataset.insert_rows(train_dataset.n_rows, labels_vec);
         std::cout << train_dataset.n_rows << std::endl;
         for (int i = 1; i < world_size; ++i){

             int random_per_datataset = random_federate_distribution();
             int n_example = train_dataset.n_cols;
             int perc_n_example = n_example * random_per_datataset / 100;

             arma::mat shuffled_train_dataset = shuffle(train_dataset, 1); // Shuffle sulle colonne (dimensione 1)
             arma::mat local_dataset = shuffled_train_dataset.cols(0, perc_n_example - 1);

             int rows = local_dataset.n_rows;
             int cols = local_dataset.n_cols;

             std::cout << "Sending matrix to "<< i <<std::endl;
             // Invia le dimensioni della matrice
             MPI_Send(&rows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
             MPI_Send(&cols, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

             // Invia i dati della matrice come un array di double
             MPI_Send(local_dataset.memptr(), rows * cols, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
             std::cout << "Master exmp for "<< i << "value = " << local_dataset.col(0).t() <<std::endl;

         }
     } else{ // client

         int rows , cols;


         // send to aggr the size of the local training set needed to compute weighted error and weight of the hyp (alpha)
         MPI_Recv(&rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         MPI_Recv(&cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

         // Crea una matrice vuota per ricevere i dati
         arma::mat local_training_set(rows, cols);

         // Ricevi i dati della matrice
         MPI_Recv(local_training_set.memptr(), rows * cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

         std::cout << "Slave "<< rank << "value = " << local_training_set.col(0).t() <<std::endl;

     }
     MPI_Finalize();

     return 0;
 }





