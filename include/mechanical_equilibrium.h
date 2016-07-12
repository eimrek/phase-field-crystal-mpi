
#ifndef MECH_EQ_H
#define MECH_EQ_H

#include "pfc.h"

#include <chrono>


typedef std::chrono::high_resolution_clock Time;


class MechanicalEquilibriumPFC : public PhaseField {
    
public:
    MechanicalEquilibriumPFC(int mpi_rank, int mpi_size);

    void test();
    double elementwise_avg_norm();

    void steepest_descent_fixed_dz();
    void steepest_descent_adaptive_dz();
    void accelerated_steepest_descent_adaptive_dz();

    double exp_line_search(double *energy_io, double **neg_direction);

    void take_step(double dz, double **neg_direction,
        complex<double> **eta_in, complex<double> **eta_out);

    void update_velocity_and_take_step(double dz, double gamma,
        double **velocity, bool zero_vel); 
};

#endif
