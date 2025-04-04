//
// Created by edoardo on 12/12/24.
//

#include "datalib.hpp"
#include <mlpack/core.hpp>

void load_datasets_and_labels(arma::mat &train_dataset, arma::Row<size_t>& train_labels, mlpack::data::DatasetInfo& info) {

    const std::string train_path = "../../datasets/covertype.train.arff";
    const std::string train_labels_path = "../../datasets/covertype.train.labels.csv";
    mlpack::data::Load(train_path, train_dataset, info, true);
    mlpack::data::Load(train_labels_path, train_labels, true);
    const arma::rowvec labels_vec = arma::conv_to<arma::rowvec>::from(arma::conv_to<arma::Row<double>>::from(train_labels));
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
    const double alpha = log((1.0 - total_error) / total_error) + log(n_class);
    if (std::isnan(alpha) || std::isinf(alpha)) {
        std::cerr << "Errore: alpha non valido" << std::endl;
        return -1 ;
    }
    return alpha;

}

void calculate_new_weights(const arma::rowvec& train_result, const double& alpha, arma::rowvec& weights){
    arma::rowvec new_result = arma::exp(alpha * (1.0 - train_result));
    auto unique_w = arma::unique(new_result);
    arma::rowvec new_weights =  new_result % weights;
    weights = new_weights / arma::sum(new_weights);
}

int index_best_model(const arma::mat & trees_error){
    arma::rowvec col_sums = arma::sum(trees_error, 0); // Somma lungo le righe (direzione verticale)
    arma::uword max_col_index = col_sums.index_max();
    return static_cast<int>(max_col_index);
}

void load_testData_and_labels(arma::mat& testDataset,arma::Row<size_t>& test_labels,  mlpack::data::DatasetInfo& info) {
    const std::string test_path = "../../datasets/covertype.test.arff";
    const std::string test_labels_path = "../../datasets/covertype.test.labels.csv";
    mlpack::data::Load(test_path, testDataset, info, true);
    mlpack::data::Load(test_labels_path, test_labels, true);
}


void accuracy_for_model(const std::vector<std::pair<mlpack::DecisionTree<>,double>>& ensemble_learning, const arma::mat& testDataset, const arma::Row<size_t>&test_labels) {
    int count = 0;
    for (const auto&[en_tree, en_alpha] : ensemble_learning) {
        arma::Row<size_t> test_predictions;
        en_tree.Classify(testDataset, test_predictions);
        auto prediction_result = arma::conv_to<arma::rowvec>::from(test_predictions == test_labels);
        double true_predictions = static_cast<double>(arma::sum(prediction_result == 1.0));
        double accuracy = true_predictions / static_cast<double>(prediction_result.n_elem);
        ++count;
    }
}

arma::Mat<size_t> predict_all_dataset(const std::vector<std::pair<mlpack::DecisionTree<>, double>>& ensemble, const arma::mat& dataset){
    arma::Mat<size_t> prediction_matrix(ensemble.size(), dataset.n_cols);
    for(int i = 0; i < ensemble.size(); ++i){
        arma::Row<size_t> predictions;
        mlpack::DecisionTree t = std::get<0>(ensemble[i]);
        t.Classify(dataset, predictions);
        prediction_matrix.row(i) = predictions;
    }
    return prediction_matrix;
}

double accuracy_ensamble(arma::Mat<size_t>prediction_matrix,const std::vector<std::pair<mlpack::DecisionTree<>, double>>& ensemble,const arma::Row<size_t>& test_labels, const int& n_class) {
    std::vector<double> final_predictions;
    for(int i = 0; i < prediction_matrix.n_cols; ++i) {
        arma::Row<size_t> models_prediction = prediction_matrix.col(i).t();

        std::vector<double> weighted_predictions(n_class, 0.0);
        for(int j = 0; j < models_prediction.n_elem; ++j){
            weighted_predictions[static_cast<int>(models_prediction(j))] += 1.0 * std::get<1>(ensemble[j]);
        }
        int most_frequent_value = -1;
        double max_count = 0;
        for (int z = 0; z < weighted_predictions.size(); ++z) {
            if (weighted_predictions[z] > max_count) {
                max_count = weighted_predictions[z];
                most_frequent_value = z;
            }

        }
        final_predictions.push_back(most_frequent_value);
    }
    const arma::rowvec p(final_predictions);
    const auto prediction_result = arma::conv_to<arma::rowvec>::from(p == test_labels);
    const double true_predictions = arma::sum((prediction_result) == 1.0) * 1.0;
    const double accuracy = true_predictions / static_cast<double>(prediction_result.n_elem);
    return accuracy;
}

int get_tree_from_majority_vote(const std::vector<int>& best_trees_index) {

    std::vector<int> vector_tree_freq(size(best_trees_index), 0);
    for (const int num : best_trees_index) {
        vector_tree_freq[num]++;
    }
    int most_frequent_value = -1;
    int max_frequency = -1;
    for (int z = 0; z < vector_tree_freq.size(); ++z) {
        if (vector_tree_freq[z] > max_frequency) {
            max_frequency = vector_tree_freq[z];
            most_frequent_value = z;
        }
    }
    return most_frequent_value;
}
