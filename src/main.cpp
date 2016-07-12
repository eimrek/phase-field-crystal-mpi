

#include <iostream>
#include <cstdlib>

#include <mpi.h>

#include "pfc.h"
#include "mechanical_equilibrium.h"

using namespace std;

void run_calculations(int mpi_rank, int mpi_size) {
    
    Time::time_point time_start = Time::now();
    Time::time_point time_var = Time::now();

    MechanicalEquilibriumPFC pfc(mpi_rank, mpi_size);
    pfc.initialize_eta();
    
    pfc.take_fft(pfc.eta_plan_f);
    
    if (mpi_rank == 0)
        printf("Initial state energy: %.16e\n", pfc.calculate_energy(pfc.eta, pfc.eta_k));

    for (int it = 0; it < 100; it++) {
        pfc.overdamped_time_step();
    }
    int it = 0;
    double simulation_time = 0.0; 
    if (mpi_rank == 0)
        printf("Initial OD energy: %.16e\n", pfc.calculate_energy(pfc.eta, pfc.eta_k));


    /*
    double en = pfc.calculate_energy(pfc.eta, pfc.eta_k);
    printf("Energy before mech eq: %.16e\n", en);
    pfc.accelerated_steepest_descent_adaptive_dz();
    */

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

