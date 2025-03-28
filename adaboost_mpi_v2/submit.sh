#!/bin/sh
#SBATCH -p broadwell
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --ntasks-per-node=18
#SBATCH -o %j.log
#SBATCH -e %j.err
#SBATCH -t 00:10:00


# Carica Spack
 . /beegfs/home/echiavazza/spack/share/spack/setup-env.sh

# Carica MPI e mlpack
spack load openmpi  
spack load mlpack

export CC=mpicc
export CXX=mpicxx

# Cartelle di build ed esecuzione
BUILD_DIR="/beegfs/home/echiavazza/unito-scpd/adaboost_mpi_v2/build"
BIN_DIR="/beegfs/home/echiavazza/unito-scpd/adaboost_mpi_v2/bin"

# Compilazione
rm -rf build/ && mkdir build && cd build
cmake ..
make -j$(nproc)

cmake .. 
make -j

# Esegui il programma con MPI dalla cartella bin
cd $BIN_DIR
srun --mpi=pmix -n 18 ./adaboost_mpi_v2
