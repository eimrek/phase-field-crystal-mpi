
#ifndef MECH_EQ_H
#define MECH_EQ_H

#include <chrono>
#include <complex>
#include <deque>

using namespace std;

typedef std::chrono::high_resolution_clock Time;

// forward declaration
class PhaseField;

class MechanicalEquilibrium {
    PhaseField *pfc;

    double elementwise_avg_norm();

    double exp_line_search(double *energy_io, double **neg_direction);

    void take_step(double dz, double **neg_direction,
        complex<double> **eta_in, complex<double> **eta_out);

    void update_velocity_and_take_step(double dz, double gamma,
        double **velocity, bool zero_vel); 

    double dot_prod(double **v1, double **v2);
    void lbfgs_direction(int m, double ***s, double ***y, double **grad, double **result);
    void move_queue(int m, double ***queue);
    bool check_move(int m, double ***q_bef, double ***q_aft);

    /** the number of lbfgs iterations before error reducing A-GD iterations*/
    static int lbfgs_iterations;

public:
    MechanicalEquilibrium(PhaseField *pfc);

    int steepest_descent_fixed_dz();
    int steepest_descent_adaptive_dz();
    int accelerated_gradient_descent(
    		double dz = 1.0,
    		int max_iter = 10000,
    		double tolerance = 7.5e-9,
    		bool print = true);
    int accelerated_gradient_descent_line_search();

    int lbfgs();
    int lbfgs_enhanced();

};

#endif
