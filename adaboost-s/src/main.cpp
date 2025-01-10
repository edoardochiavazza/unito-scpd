#include <iostream>
#include <mlpack.hpp>
#include <vector>

double calculate_total_error(const arma::rowvec& train_result, const arma::rowvec& weights){
    double total_error = arma::sum((1.0 - train_result) % weights);
    std::cout << total_error<< std::endl;
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

arma::Mat<size_t> predict_all_dataset(const std::vector<std::pair<mlpack::DecisionTree<>, double>>& ensemble, const arma::mat& dataset){
    arma::Mat<size_t> prediction_matrix(ensemble.size(), dataset.n_cols);
    for(int i = 0; i < ensemble.size(); ++i){
        arma::Row<size_t> predictions;
        mlpack::DecisionTree t = std::get<0>(ensemble[i]);
        t.Classify(dataset, predictions);
        std::cout<<"Classification start from "<< i << std::endl;
        prediction_matrix.row(i) = predictions;
    }
    return prediction_matrix;
}

void accuracy_ensamble(arma::Mat<size_t>prediction_matrix,const std::vector<std::pair<mlpack::DecisionTree<>, double>>& ensemble,const arma::Row<size_t>& test_labels) {

    std::cout<<"Weighted vote start"<< std::endl;
    std::vector<double> final_predictions;

    for(int i = 0; i < prediction_matrix.n_cols; ++i) {
        arma::Row<size_t> models_prediction = prediction_matrix.col(i).t();
        std::unordered_map<int, double> frequency_map;
        // Conta le occorrenze dei valori nella colonna
        std::cout<<"models_prediction.n_elem = "<< models_prediction.n_elem <<std::endl;
        for(int j = 0; j < models_prediction.n_elem; ++j){
            std::cout<<"model n: "<< j << " alpha = "<<std::get<1>(ensemble[j]) <<" prediction: " << models_prediction(j)<< std::endl;
            frequency_map[static_cast<int>(models_prediction(j))] += 1.0 * std::get<1>(ensemble[j]);
        }
        int most_frequent_value = -1;
        double max_count = 0;
        for (const auto &[class_num, frequency]: frequency_map) {
            if (frequency > max_count) {
                max_count = frequency;
                most_frequent_value = class_num;
            }

        }
        final_predictions.push_back(most_frequent_value);
    }

    arma::rowvec p(final_predictions);

    auto prediction_result = arma::conv_to<arma::rowvec>::from(p == test_labels);
    double true_predictions = arma::sum((prediction_result) == 1.0) * 1.0;
    double accuracy = true_predictions / static_cast<double>(prediction_result.n_elem);
    std::cout<<"Accuracy = " << accuracy << std::endl;
}

void accuracy_for_model(const std::vector<std::pair<mlpack::DecisionTree<>,double>>& ensemble_learning, const arma::mat& testDataset, const arma::Row<size_t>&test_labels) {
    int count = 0;
    for (const auto&[en_tree, en_alpha] : ensemble_learning) {
        arma::Row<size_t> test_predictions;
        en_tree.Classify(testDataset, test_predictions);
        auto prediction_result = arma::conv_to<arma::rowvec>::from(test_predictions == test_labels);
        double true_predictions = static_cast<double>(arma::sum(prediction_result == 1.0));
        double accuracy = true_predictions / static_cast<double>(prediction_result.n_elem);
        std::cout<<"Tree trained in epoch = " << count << " Accuracy = " << accuracy << std::endl;
        ++count;
    }
}

int main() {

    constexpr int n_model = 3;
    //Define path of test,train dataset and label as constexpr
    const std::string train_path = "../datasets/covertype.train.arff";
    const std::string train_labels_path = "../datasets/covertype.train.labels.csv";
    const std::string test_path = "../datasets/covertype.test.arff";
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
    std::cout << "exmp n " <<trainDataset.n_cols << std::endl;
    std::cout << "feature n " <<trainDataset.n_rows << std::endl;
// Add weights with value of 1/len(dataset) for implementing adaboost

    std::cout << "Create weights" << std::endl;

    arma::rowvec weights(trainDataset.n_cols, arma::fill::ones);
    weights /= static_cast<double>(trainDataset.n_cols); //2.4588e-06
    arma::Row<size_t> unique_labels = arma::unique(train_labels);
    int n_class = static_cast<int>(unique_labels.n_elem);

// Create pair vector for store (model, alpha). Alpha rappresent the accuracy of the model.
    std::vector<std::pair<mlpack::DecisionTree<>, double>> ensemble;

    for(int i=0; i < n_model; ++i){
        arma::Row<size_t> predictions;
        std::cout << "Train tree n: " << i <<std::endl;
        mlpack::DecisionTree tree(trainDataset, info, train_labels, unique_labels.size(), weights, 10, 1e-7, 10);
        tree.Classify(trainDataset, predictions);
        arma::rowvec train_result = arma::conv_to<arma::rowvec>::from(predictions == train_labels);
        double total_error = calculate_total_error(train_result,weights);
        double alpha = calculate_alpha(total_error,n_class);
        ensemble.emplace_back(tree, alpha);
        calculate_new_weights(train_result, alpha, weights);
    }
    accuracy_for_model(ensemble,testDataset,test_labels);
    arma::Mat<size_t> prediction_matrix = predict_all_dataset(ensemble, testDataset);
    accuracy_ensamble(prediction_matrix,ensemble,test_labels);
    return 0;

}

