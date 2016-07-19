
#ifndef MECH_EQ_H
#define MECH_EQ_H

#include <chrono>
#include <complex>

using namespace std;

typedef std::chrono::high_resolution_clock Time;

// forward declaration
class PhaseField;

class MechanicalEquilibrium {
    PhaseField *pfc;
public:
    MechanicalEquilibrium(PhaseField *pfc);

    double elementwise_avg_norm();

    int steepest_descent_fixed_dz();
    int steepest_descent_adaptive_dz();
    int accelerated_steepest_descent_adaptive_dz(int *p_n_fft);

    double exp_line_search(double *energy_io, double **neg_direction, int *p_n_fft);

    void take_step(double dz, double **neg_direction,
        complex<double> **eta_in, complex<double> **eta_out);

    void update_velocity_and_take_step(double dz, double gamma,
        double **velocity, bool zero_vel); 

};

#endif
