 // needed only for using library routines
 //
# include <mpi.h>
#include <cstdio>

 int main(int argc, char ** argv)
 {
     int rank, size;
     MPI_Init(&argc, &argv);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
     printf("I am %d of %d\n", rank + 1, size);
     MPI_Finalize();
     return 0;
 }





