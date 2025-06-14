//
// Created by edoardo on 12/12/24.
//

#include <mpi.h>
#include "libcomm.hpp"
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <cereal/archives/binary.hpp>  // Include Cereal

void send_data_to_client(const arma::mat &local_dataset,const int &client_rank) {
    const int rows = static_cast<int>(local_dataset.n_rows);
    const int cols = static_cast<int>(local_dataset.n_cols);

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

}


mlpack::DecisionTree<> receive_tree(const int& rank)
{

    size_t length;
    MPI_Recv(&length, 1, MPI_UNSIGNED_LONG, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<char> buffer(length);

    MPI_Recv(buffer.data(), static_cast<int>(length), MPI_BYTE, rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mlpack::DecisionTree<> tree = deserialize_tree(std::string(buffer.begin(), buffer.end()));

    buffer.clear();
    return tree;
}


 mlpack::DecisionTree<> broadcast_tree(mlpack::DecisionTree<> &tree, MPI_Comm comm)
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

std::vector<mlpack::DecisionTree<>> gather_tree(mlpack::DecisionTree<> &tree, const int &rank, const int& world_size) {

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