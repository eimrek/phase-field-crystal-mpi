

#include <iostream>

#include <mpi.h>

#include "pfc.h"
#include "mechanical_equilibrium.h"


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    
    {
        
        PhaseField phase_field(mpi_size, mpi_rank);
        phase_field.initialize_eta();
        phase_field.take_fft(phase_field.plan_forward);
        phase_field.calculate_energy();
        phase_field.overdamped_time_step();
        phase_field.calculate_energy();

        phase_field.output_field(phase_field.eta[0]);

        MechanicalEquilibrium mech_eq(&phase_field);
        mech_eq.test();
    }

    MPI_Finalize();
}

