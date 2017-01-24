

#include <iostream>
#include <cstdlib>

#include <mpi.h>

#include "pfc.h"

using namespace std;

void run_calculations(int mpi_rank, int mpi_size) {

    PhaseField pfc(mpi_rank, mpi_size);

    //pfc.start_calculations();
    pfc.test();

}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    cout << "Process started: " << mpi_rank << "/" << mpi_size << endl;

    run_calculations(mpi_rank, mpi_size);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

