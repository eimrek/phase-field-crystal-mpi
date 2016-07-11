
#include <iostream>
#include <chrono>

#include <mpi.h>

#include "mechanical_equilibrium.h"

typedef std::chrono::high_resolution_clock Time;

MechanicalEquilibrium::MechanicalEquilibrium(int mpi_size, int mpi_rank)
        : PhaseField(mpi_size, mpi_rank) {
}


void MechanicalEquilibrium::test() {
    std::cout << "M.E. test" << std::endl;
}

/*! 
 *  Method, which will take the elementwise 1st order norm
 *  of the gradient, which is assumed to be in "grad_theta"
 */
double MechanicalEquilibrium::elementwise_avg_norm() {
    double local_norm = 0.0;
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int c = 0; c < nc; c++) {
               local_norm += abs(grad_theta[c][i*ny + j]); 
            }
        }
    }
    double norm = 0.0;
    MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return norm/(3*nx*ny);
}


void MechanicalEquilibrium::take_step(double dz, double **neg_direction,
        complex<double> **eta_in, complex<double> **eta_out) {
    
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int c = 0; c < nc; c++) {
                double dtheta = -dz*neg_direction[c][i*ny + j];
                eta_out[c][i*ny + j] = eta_in[c][i*ny + j]
                                       * exp(complex<double>(0.0,1.0)*dtheta); 
            }
        }
    }
}


void MechanicalEquilibrium::steepest_descent_fixed_dz() {
    double dz = 1.0;
    int max_iter = 10000;
    int check_freq = 100;
    double tolerance = 7.5e-9;

    // update eta_k (just in case)
    take_fft(eta_plan_f);

    double last_energy = calculate_energy(eta, eta_k);

    Time::time_point time_var = Time::now();

    for (int it = 1; it <= max_iter; it++) {
        // First, calculate the gradient (will be in "buffer")
        // NB: eta_k needs to be set
        calculate_grad_theta(eta, eta_k);
        
        take_step(dz, grad_theta, eta, eta);

        // update eta_k 
        take_fft(eta_plan_f);

        if (it % check_freq == 0) {
            double energy = calculate_energy(eta, eta_k);
            double error = elementwise_avg_norm();
            if (mpi_rank == 0) {
                double dur = std::chrono::duration<double>(Time::now()-time_var).count();
                time_var = Time::now();
                printf("it: %5d; energy: %.16e; err: %.16e; time: %4.1f\n", it, energy,
                        error, dur);
                if (energy > last_energy) {
                    printf("NB: energy increased!\n");
                }
            }
            last_energy = energy;
            if (error < tolerance) {
                printf("Solution found\n");
                break;
            }
        }
    }

}

/*! Exponential line search method
 *
 *  Finds first suitable step by trying expoentially increasing steps
 *  Will also set "eta" and "eta_k"
 *
 *  @param energy_io input: starting energy; output: energy of the taken step
 *  @return step size
 */
double MechanicalEquilibrium::exp_line_search(double *energy_io, double **neg_direction) {
    double dz_start = 1.0;
    double search_factor = 2.0;

    int largest_step_power = 20;
    int smallest_step_power = 6;
    
    // Allocate memory to hold saved eta values (no need for FFT plans)
    complex<double> **eta_prev = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    complex<double> **eta_prev_k = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    for (int i = 0; i < nc; i++) {
        eta_prev[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
        eta_prev_k[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
    }

    // Take initial step and store result to eta_tmp
    take_step(dz_start, neg_direction, eta, eta_tmp);
    take_fft(eta_tmp_plan_f);
    double energy = calculate_energy(eta_tmp, eta_tmp_k);

    if (energy < *energy_io) {
        // save the successful step
        // (in case next is worse, so it will be taken)
        memcopy_eta(eta_prev, eta_tmp);
        memcopy_eta(eta_prev_k, eta_tmp_k);
    } else {
        // search smaller steps
        search_factor = 1.0/search_factor;
    }

    //printf("dz: %4.2f; en: %.16e\n", dz_start, energy);
    double last_energy = energy;

    double dz = dz_start;
    for (int t = 0; t < largest_step_power; t++) {
        // try to increase step size
        dz *= search_factor;

        take_step(dz, neg_direction, eta, eta_tmp);
        take_fft(eta_tmp_plan_f);
        double energy = calculate_energy(eta_tmp, eta_tmp_k);

        //printf("dz: %4.2f; en: %.16e\n", dz, energy);

        // If we're searching bigger steps, take longest step, which
        // decreases the energy
        if (search_factor > 1.0) {
            if (energy < last_energy) {
                // save this step result and continue
                memcopy_eta(eta_prev, eta_tmp);
                memcopy_eta(eta_prev_k, eta_tmp_k);
            } else {
                // the previous step is chosen.
                *energy_io = last_energy;
                memcopy_eta(eta, eta_prev);
                memcopy_eta(eta_k, eta_prev_k);
                dz = dz/search_factor;
                break;
            }
        } else {
            // If searching smaller steps, take first one that decreases
            // the energy wrt starting energy 
            if (energy < *energy_io) {
                memcopy_eta(eta, eta_tmp);
                memcopy_eta(eta_k, eta_tmp_k);
                *energy_io = energy;
                break;
            }
            if (t+1 >= smallest_step_power) {
                cout << "Warning: didn't find step." << endl;
                dz = 0.0;
                break;
            }
        }
        last_energy = energy;
        if (t == largest_step_power-1) {
            *energy_io = energy;
            cout << "Warning: longest step limit reached." << endl;
        }
    }
    
    for (int i = 0; i < nc; i++) {
        fftw_free(eta_prev[i]);
        fftw_free(eta_prev_k[i]);
    }
    free(eta_prev); free(eta_prev_k);

    return dz;
}

void MechanicalEquilibrium::steepest_descent_adaptive_dz() {
    int max_iter = 10000;
    double tolerance = 7.5e-9;

    // update eta_k (just in case)
    take_fft(eta_plan_f);

    double energy = calculate_energy(eta, eta_k);
    double last_energy = energy;

    // Calculate the gradient
    // NB: eta_k needs to be set
    calculate_grad_theta(eta, eta_k);

    Time::time_point time_var = Time::now();

    for (int it = 1; it <= max_iter; it++) {
        // Do the exponential line search to find optimal step
        // will update eta, eta_k and also store new energy value
        double dz = exp_line_search(&energy, grad_theta);

        // for this iteration's error check and next iteration's step
        calculate_grad_theta(eta, eta_k);
        double error = elementwise_avg_norm();

        if (mpi_rank == 0) {
            double dur = std::chrono::duration<double>(Time::now()-time_var).count();
            time_var = Time::now();
            printf("it: %5d; dz: %6.2f; energy: %.16e; err: %.16e; time: %4.1f\n", it, dz, 
                    energy, error, dur); 
            if (energy > last_energy) {
                printf("NB: energy increased!\n");
            }
        }
        last_energy = energy;

        if (error < tolerance) {
            printf("Solution found!\n");
            break;
        }
    }
}


void MechanicalEquilibrium::update_velocity_and_take_step(double dz, double gamma,
        double **velocity, bool zero_vel) {
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int c = 0; c < nc; c++) {
                if (zero_vel) 
                    velocity[c][i*ny + j] = dz * grad_theta[c][i*ny + j];
                else
                    velocity[c][i*ny + j] = gamma*velocity[c][i*ny+j]
                                            + dz*grad_theta[c][i*ny + j];
                eta[c][i*ny + j] *= exp(complex<double>(0.0, 1.0)
                        *(-1.0)*velocity[c][i*ny + j]);
            }
        }
    }
}


void MechanicalEquilibrium::accelerated_steepest_descent_adaptive_dz() {
    double dz_accd = 1.0;
    int max_iter = 10000;
    double tolerance = 7.5e-9;

    int adaptive_step_freq = 100;
    int num_adaptive_steps = 5;
    int check_freq = 50;

    double gamma = 0.9;
    
    Time::time_point time_start = Time::now();
    Time::time_point time_var = time_start;

    // Allocate memory to hold velocity values (no need for FFT plans)
    // Note that the actual steps will be taken in negative direction of velocity
    double **velocity= (double **) malloc(sizeof(double*)*nc);
    for (int i = 0; i < nc; i++)
        velocity[i] = (double*) malloc(sizeof(double)*local_nx*ny);

    // Boolean when to ignore velocity (first iteration and after adaptive steps)
    bool zero_velocity = true;

    double last_energy = calculate_energy(eta, eta_k);
    // update eta_k 
    take_fft(eta_plan_f);

    int it = 1;
    while (it <= max_iter) {
        if ((it-1) % adaptive_step_freq == 0) {
            // -----------------------------------------------
            // Run the specified number of adaptive steps
            for (int sn = 0; sn < num_adaptive_steps; sn++) {
                calculate_grad_theta(eta, eta_k);

                double energy_io = calculate_energy(eta, eta_k);
                double dz = exp_line_search(&energy_io, grad_theta);
                
                double error = elementwise_avg_norm();

                if (mpi_rank == 0)
                    printf("it: %5d; adaptive step: %6.1f; energy: %.16e; err: %.16e\n",
                            it, dz, energy_io, error);

                last_energy = energy_io;
                it++;
            }
            zero_velocity = true;
            // -----------------------------------------------
        }
        // Resume with accelerated descent
        if (zero_velocity) {
            // If last step velocity is zero (or uninitialized) take normal gradient
            calculate_grad_theta(eta, eta_k);

        } else {
            // If last step velocity is not zero, take a prediction gradient
            take_step(gamma, velocity, eta, eta_tmp);
            // update eta_tmp_k
            take_fft(eta_tmp_plan_f);
            // calculate gradient based on eta_tmp_k
            calculate_grad_theta(eta_tmp, eta_tmp_k);
        }
        update_velocity_and_take_step(dz_accd, gamma, velocity, zero_velocity);
        take_fft(eta_plan_f);
        zero_velocity = false;

        if (it % check_freq == 0) {
            double energy = calculate_energy(eta, eta_k);
            double error = elementwise_avg_norm();
            if (mpi_rank == 0) {
                // timings ---------
                double it_dur = std::chrono::duration<double>(Time::now()-time_var).count();
                double tot_dur = std::chrono::
                    duration<double>(Time::now()-time_start).count();
                time_var = Time::now();
                // -----------------
                printf("it: %5d; energy: %.16e; err: %.16e; time: %4.1f; tot_time: %6.1f\n",
                        it, energy, error, it_dur, tot_dur);
                if (energy > last_energy) cout << "Warning: energy increased." << endl;
                if (error < tolerance) cout << "Solution found." << endl;
            }
            last_energy = energy;
            if (error < tolerance) break;
        }
        if (it >= max_iter)
            printf("Solution was not found within %d iterations.\n", max_iter);
        it++;
    }

    for (int i = 0; i < nc; i++)
        fftw_free(velocity[i]);
    free(velocity);
}

