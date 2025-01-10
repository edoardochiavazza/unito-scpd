//
// Created by edoardo on 12/12/24.
//

#include "datalib.hpp"
#include <mlpack/core.hpp>

void load_datasets_and_labels(arma::mat &train_dataset, arma::Row<size_t>& train_labels, mlpack::data::DatasetInfo& info) {

    const std::string train_path = "/home/edoardo/Desktop/unito-scpd/adaboost-mpi/datasets/covertype.train.arff";
    const std::string train_labels_path = "/home/edoardo/Desktop/unito-scpd/adaboost-mpi/datasets/covertype.train.labels.csv";
    mlpack::data::Load(train_path, train_dataset, info, true);
    mlpack::data::Load(train_labels_path, train_labels, true);
    arma::rowvec labels_vec = arma::conv_to<arma::rowvec>::from(arma::conv_to<arma::Row<double>>::from(train_labels));
    train_dataset.insert_rows(train_dataset.n_rows, labels_vec);
}

double calculate_total_error(const arma::rowvec& train_result, const arma::rowvec& weights){
    double total_error = arma::sum((1.0 - train_result) % weights);
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

int index_best_model(const arma::mat & trees_error){

    arma::rowvec col_sums = arma::sum(trees_error, 0); // Somma lungo le righe (direzione verticale)

    arma::uword max_col_index = col_sums.index_max();

    return static_cast<int>(max_col_index);
}