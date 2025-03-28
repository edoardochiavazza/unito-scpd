#!/bin/sh
#SBATCH -p broadwell
#SBATCH --nodes=10
#SBATCH --ntasks=330
#SBATCH --ntasks-per-node=33
#SBATCH -o %j.log
#SBATCH -e %j.err
#SBATCH -t 01:00:00
#SBATCH --job-name=adab_v1_13

# Carica Spack
 . /beegfs/home/echiavazza/spack/share/spack/setup-env.sh

# Carica MPI e mlpack
spack load openmpi  
spack load mlpack

export CC=mpicc
export CXX=mpicxx

# Cartelle di build ed esecuzione
BUILD_DIR="/beegfs/home/echiavazza/unito-scpd/adaboost-mpi"
BIN_DIR="/beegfs/home/echiavazza/unito-scpd/adaboost-mpi/bin"

# Compilazione
cd $BUILD_DIR
rm -rf build/ && mkdir build && cd build
cmake .. 
make -j

# Esegui il programma con MPI dalla cartella bin
cd $BIN_DIR
srun --mpi=pmix -n 330 ./adaboost-mpi 10 33
