

#include <iostream>
#include <cstdlib>

#include <mpi.h>

#include "pfc.h"
#include "mechanical_equilibrium.h"


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    
    cout << "Process started: " << mpi_rank << "/" << mpi_size << endl;
    {
        MechanicalEquilibrium pfc(mpi_size, mpi_rank);
        pfc.initialize_eta();
        pfc.output_field(pfc.eta[1]);
        pfc.write_eta_to_file();
        
        pfc.take_fft(pfc.eta_plan_f);
        for (int it = 0; it < 80; it++) {
            pfc.overdamped_time_step();
        }
        pfc.output_field(pfc.eta[1]);
        pfc.read_eta_from_file();
        pfc.output_field(pfc.eta[1]);

        /*
        double en = pfc.calculate_energy(pfc.eta, pfc.eta_k);
        printf("Energy before mech eq: %.16e\n", en);
        pfc.accelerated_steepest_descent_adaptive_dz();
        */

    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

