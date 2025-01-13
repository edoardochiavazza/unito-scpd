#include <iostream>
#include <mlpack.hpp>
#include <vector>

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
    auto unique_w = arma::unique(new_result);
    arma::rowvec new_weights =  new_result % weights;
    weights = (new_weights / arma::sum(new_weights));
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

double accuracy_ensamble(arma::Mat<size_t>prediction_matrix,const std::vector<std::pair<mlpack::DecisionTree<>, double>>& ensemble,const arma::Row<size_t>& test_labels, std::string dataset_type) {
    std::vector<double> final_predictions;

    for(int i = 0; i < prediction_matrix.n_cols; ++i) {
        arma::Row<size_t> models_prediction = prediction_matrix.col(i).t();
        std::unordered_map<int, double> frequency_map;
        // Conta le occorrenze dei valori nella colonna
        for(int j = 0; j < models_prediction.n_elem; ++j){
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
    std::cout<<"Ensamble accuracy = " << accuracy <<" for " << dataset_type<<" dataset"<<std::endl;
    return accuracy;
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

    constexpr int n_model = 40;
    int epochs[7] = { 5,10,20,30,40,50,100};
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
// Add weights with value of 1/len(dataset) for implementing adaboost
    for(auto e : epochs){
        std::cout << "Create weights" << std::endl;

        arma::rowvec weights(trainDataset.n_cols, arma::fill::ones);
        weights /= static_cast<double>(trainDataset.n_cols); //2.4588e-06
        arma::Row<size_t> unique_labels = arma::unique(train_labels);
        int n_class = static_cast<int>(unique_labels.n_elem);

// Create pair vector for store (model, alpha). Alpha rappresent the accuracy of the model.
        std::vector<std::pair<mlpack::DecisionTree<>, double>> ensemble;
        double average_time_sequential_epoch = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i < e; ++i){
            double time_sequential_epoch = 0;
            auto start_epoch = std::chrono::high_resolution_clock::now();
            arma::Row<size_t> predictions;
            mlpack::DecisionTree tree(trainDataset, info, train_labels, unique_labels.size(), weights, 10, 1e-7, 10);
            tree.Classify(trainDataset, predictions);
            arma::rowvec train_result = arma::conv_to<arma::rowvec>::from(predictions == train_labels);
            double total_error = calculate_total_error(train_result,weights);
            double alpha = calculate_alpha(total_error,n_class);
            ensemble.emplace_back(tree, alpha);
            calculate_new_weights(train_result, alpha, weights);
            auto end_epoch_timer = std::chrono::high_resolution_clock::now();
            time_sequential_epoch =  std::chrono::duration<double>(end_epoch_timer - start_epoch).count();
            average_time_sequential_epoch = (average_time_sequential_epoch + time_sequential_epoch)/2;
            std::cout << "Epoch " << i << " end in " << time_sequential_epoch << " seconds\n";
        }
        auto end_total_timer = std::chrono::high_resolution_clock::now();
        double time_sequential_total = std::chrono::duration<double>(end_total_timer - start).count();
        std::cout << "Time epoch (T1): " << average_time_sequential_epoch << " seconds\n";
        std::cout << "Time epochs (T1): " << time_sequential_total << " seconds\n";
        accuracy_for_model(ensemble,testDataset,test_labels);
        arma::Mat<size_t> prediction_matrix_test = predict_all_dataset(ensemble, testDataset);
        arma::Mat<size_t> prediction_matrix_training = predict_all_dataset(ensemble, trainDataset);
        double acc_ensable_test = accuracy_ensamble(prediction_matrix_test,ensemble,test_labels, "test");
        double acc_ensable_training  = accuracy_ensamble(prediction_matrix_training ,ensemble,train_labels, "training");
        // Nome del file di output
        std::string fileName = "risultati_adaboost-s.txt";

        // Creazione di un oggetto di tipo ofstream
        std::ofstream outputFile(fileName, std::ios::app);

        // Controllo che il file sia stato aperto correttamente
        if (!outputFile.is_open()) {
            std::cerr << "Errore nell'apertura del file: " << fileName << std::endl;
            return 1;
        }

        // Scrittura dei risultati nel file

        outputFile << "--------------------------\n";
        outputFile << "Machine: Macbook\n";
        outputFile << "Number epoch: "<< e<<"\n";
        outputFile << "Time epoch (T1): " << average_time_sequential_epoch << " seconds\n";
        outputFile << "Time epochs (T1): " << time_sequential_total << " seconds\n";
        outputFile << "Ensamble accuracy = " << acc_ensable_test <<" for test dataset\n";
        outputFile << "Ensamble accuracy = " << acc_ensable_training <<" for training dataset\n";
        // Chiusura del file
        outputFile.close();
        std::cout << "Risultati scritti con successo nel file: " << fileName <<"per il valore " << e<<std::endl;
    }
    return 0;
}


