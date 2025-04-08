#!/bin/sh
#SBATCH -p broadwell
#SBATCH --nodes=9
#SBATCH --ntasks=324
#SBATCH --ntasks-per-node=36
#SBATCH -o %j.log
#SBATCH -e %j.err
#SBATCH -t 06:00:00


# Carica Spack
 . /beegfs/home/echiavazza/spack/share/spack/setup-env.sh

# Carica MPI e mlpack
spack load openmpi  
spack load mlpack
spack load cmake

#export CC=mpicc
#export CXX=mpicxx

cd /beegfs/home/echiavazza/unito-scpd/adaboost_mpi_v2
# Crea o pulisci le directory bin e build
mkdir -p bin
mkdir -p build
rm -rf build/*

# Compilazione
cd build
cmake ..
make -j $(nproc)

# Esegui il programma con MPI dalla cartella bin
cd ../bin

srun --mpi=pmix --verbose -n 324 ./adaboost_mpi_v2 9 36
