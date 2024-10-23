 // needed only for using library routines
 //
#include <mpi.h>
#include <cstdio>
#include <mlpack.hpp>
#include <array>

int random_federate_distribution(){

    //generate a number between 70 and 100, that will be the percentage of dataset own by the client
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> random_distribution_number(70, 101);
    int random_distr = random_distribution_number(gen);
    return  random_distr;

}

int select_random_example(int n_example){
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> random_distribution_example(0, n_example + 1);
    int index_random_example = random_distribution_example(gen);
    return index_random_example;
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
         const std::string train_path = "/Users/edoardochiavazza/CLionProjects/adaboost-mpi/datasets/covertype.train.arff";
         const std::string train_labels_path = "/Users/edoardochiavazza/CLionProjects/adaboost-mpi/datasets/covertype.train.labels.csv";
         mlpack::data::Load(train_path, train_dataset, info, true);
         mlpack::data::Load(train_labels_path, train_labels, true);

         for (int i = 0; i < world_size;++i){

             arma::mat local_trainig_dataset(train_dataset.n_rows, train_dataset.n_cols);
             int random_per_datataset = random_federate_distribution();
             int n_example = train_dataset.n_cols;
             int perc_n_example = n_example * random_per_datataset / 100;

             arma::uvec indices = arma::linspace<arma::uvec>(0, n_example - 1, n_example);

             // Rimescolo gli indici
             arma::uvec shuffled_indices = arma::shuffle(indices);

             // Seleziono l'80% delle colonne
             arma::uvec selected_indices = shuffled_indices.head(perc_n_example);

             for(int k = 0; k < selected_indices.n_elem; ++k){
                local_trainig_dataset.insert_cols()
             }
             // Creo la nuova matrice con l'80% delle colonne
             arma::mat local_trainig_dataset = train_dataset.cols(selected_indices);

             // Creo un nuovo vettore di label con l'80% degli elementi
             arma::Row<size_t> local_trainig_labels = train_labels(selected_indices);

             std::cout<<"Esempio 0 non shuffle"<<train_dataset(0)<<std::endl;
             std::cout<<"Esempio 0 shuffle"<<local_trainig_dataset(0)<<std::endl;
         }
     } else{ // client
         // send to aggr the size of the local training set needed to compute weighted error and weight of the hyp (alpha)


     }

     MPI_Finalize();
         //Define path of test,train dataset and label as constexpr
         /*
         const std::string train_path = "../datasets/covertype.train.arff";
         const std::string train_labels_path = "../datasets/covertype.train.labels.csv";
         const std::string test_path = "/Users/edoardochiavazza/CLionProjects/adaboost-mpi/datasets/covertype.test.arff";
         const std::string test_labels_path = "../datasets/covertype.test.labels.csv";

         // Load a categorical dataset.
         arma::mat trainDataset;
         arma::mat testDataset;
         mlpack::data::DatasetInfo info;
         arma::Row<size_t> train_labels; // the labels are 0,1,2,3,4,5,6 represent one of forest's type
         arma::Row<size_t> test_labels;

         std::cout << "Loading " << train_path << std::endl;
         std::cout << "Loading " << train_labels_path << std::endl;


         mlpack::data::Load(test_path, testDataset, info, true);
         mlpack::data::Load(test_labels_path, test_labels, true);


         mlpack::data::Load(train_path, trainDataset, info, true);
         mlpack::data::Load(train_labels_path, train_labels, true);
     }
          */
     return 0;
 }





