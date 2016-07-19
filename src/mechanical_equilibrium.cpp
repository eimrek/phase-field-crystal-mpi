
#include <iostream>

#include <mpi.h>

#include "mechanical_equilibrium.h"

#include "pfc.h"


MechanicalEquilibrium::MechanicalEquilibrium(PhaseField *pfc)
        : pfc(pfc) {}


/*! 
 *  Method, which will take the elementwise 1st order norm
 *  of the gradient, which is assumed to be in "grad_theta"
 */
double MechanicalEquilibrium::elementwise_avg_norm() {
    double local_norm = 0.0;
    for (int i = 0; i < pfc->local_nx; i++) {
        for (int j = 0; j < pfc->ny; j++) {
            for (int c = 0; c < pfc->nc; c++) {
               local_norm += abs(pfc->grad_theta[c][i*pfc->ny + j]); 
            }
        }
    }
    double norm = 0.0;
    MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return norm/(3*pfc->nx*pfc->ny);
}


void MechanicalEquilibrium::take_step(double dz, double **neg_direction,
        complex<double> **eta_in, complex<double> **eta_out) {
    
    for (int i = 0; i < pfc->local_nx; i++) {
        for (int j = 0; j < pfc->ny; j++) {
            for (int c = 0; c < pfc->nc; c++) {
                double dtheta = -dz*neg_direction[c][i*pfc->ny + j];
                eta_out[c][i*pfc->ny + j] = eta_in[c][i*pfc->ny + j]
                                       * exp(complex<double>(0.0,1.0)*dtheta); 
            }
        }
    }
}


int MechanicalEquilibrium::steepest_descent_fixed_dz() {
    double dz = 1.0;
    int max_iter = 10000;
    int check_freq = 100;
    double tolerance = 7.5e-9;

    // update eta_k (just in case)
    pfc->take_fft(pfc->eta_plan_f);

    double last_energy = pfc->calculate_energy(pfc->eta, pfc->eta_k);

    Time::time_point time_var = Time::now();
    
    int it = 1;
    for (; it <= max_iter; it++) {
        // First, calculate the gradient (will be in "buffer")
        // NB: eta_k needs to be set
        pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);
        
        take_step(dz, pfc->grad_theta, pfc->eta, pfc->eta);

        // update eta_k 
        pfc->take_fft(pfc->eta_plan_f);

        if (it % check_freq == 0) {
            double energy = pfc->calculate_energy(pfc->eta, pfc->eta_k);
            double error = elementwise_avg_norm();
            if (pfc->mpi_rank == 0) {
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
    return it;
}

/*! Exponential line search method
 *
 *  Finds first suitable step by trying expoentially increasing steps
 *  Will also set "eta" and "eta_k"
 *
 *  @param energy_io input: starting energy; output: energy of the taken step
 *  @return step size
 */
double MechanicalEquilibrium::exp_line_search(double *energy_io, double **neg_direction,
        int *p_n_fft) {
    double dz_start = 1.0;
    double search_factor = 2.0;

    int largest_step_power = 20;
    int smallest_step_power = 6;
    
    // Allocate memory to hold saved eta values (no need for FFT plans)
    complex<double> **eta_prev = (complex<double>**)
        malloc(sizeof(complex<double>*)*pfc->nc);
    complex<double> **eta_prev_k = (complex<double>**)
        malloc(sizeof(complex<double>*)*pfc->nc);
    for (int i = 0; i < pfc->nc; i++) {
        eta_prev[i] = reinterpret_cast<complex<double>*>
            (fftw_alloc_complex(pfc->alloc_local));
        eta_prev_k[i] = reinterpret_cast<complex<double>*>
            (fftw_alloc_complex(pfc->alloc_local));
    }

    // Take initial step and store result to eta_tmp
    take_step(dz_start, neg_direction, pfc->eta, pfc->eta_tmp);
    pfc->take_fft(pfc->eta_tmp_plan_f); (*p_n_fft)++;
    double energy = pfc->calculate_energy(pfc->eta_tmp, pfc->eta_tmp_k); (*p_n_fft)++;

    if (energy < *energy_io) {
        // save the successful step
        // (in case next is worse, so it will be taken)
        pfc->memcopy_eta(eta_prev, pfc->eta_tmp);
        pfc->memcopy_eta(eta_prev_k, pfc->eta_tmp_k);
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

        take_step(dz, neg_direction, pfc->eta, pfc->eta_tmp);
        pfc->take_fft(pfc->eta_tmp_plan_f); (*p_n_fft)++;
        double energy = pfc->calculate_energy(pfc->eta_tmp, pfc->eta_tmp_k); (*p_n_fft)++;

        //printf("dz: %4.2f; en: %.16e\n", dz, energy);

        // If we're searching bigger steps, take longest step, which
        // decreases the energy
        if (search_factor > 1.0) {
            if (energy < last_energy) {
                // save this step result and continue
                pfc->memcopy_eta(eta_prev, pfc->eta_tmp);
                pfc->memcopy_eta(eta_prev_k, pfc->eta_tmp_k);
            } else {
                // the previous step is chosen.
                *energy_io = last_energy;
                pfc->memcopy_eta(pfc->eta, eta_prev);
                pfc->memcopy_eta(pfc->eta_k, eta_prev_k);
                dz = dz/search_factor;
                break;
            }
        } else {
            // If searching smaller steps, take first one that decreases
            // the energy wrt starting energy 
            if (energy < *energy_io) {
                pfc->memcopy_eta(pfc->eta, pfc->eta_tmp);
                pfc->memcopy_eta(pfc->eta_k, pfc->eta_tmp_k);
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
    
    for (int i = 0; i < pfc->nc; i++) {
        fftw_free(eta_prev[i]);
        fftw_free(eta_prev_k[i]);
    }
    free(eta_prev); free(eta_prev_k);

    return dz;
}

int MechanicalEquilibrium::steepest_descent_adaptive_dz() {
    int max_iter = 10000;
    double tolerance = 7.5e-9;

    // update eta_k (just in case)
    pfc->take_fft(pfc->eta_plan_f);

    double energy = pfc->calculate_energy(pfc->eta, pfc->eta_k);
    double last_energy = energy;

    // Calculate the gradient
    // NB: eta_k needs to be set
    pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);

    Time::time_point time_var = Time::now();

    int it = 1;
    for (; it <= max_iter; it++) {
        // Do the exponential line search to find optimal step
        // will update eta, eta_k and also store new energy value
        int num_fft = 0;
        double dz = exp_line_search(&energy, pfc->grad_theta, &num_fft);

        // for this iteration's error check and next iteration's step
        pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);
        double error = elementwise_avg_norm();

        if (pfc->mpi_rank == 0) {
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

    return it;
}


void MechanicalEquilibrium::update_velocity_and_take_step(double dz, double gamma,
        double **velocity, bool zero_vel) {
    for (int i = 0; i < pfc->local_nx; i++) {
        for (int j = 0; j < pfc->ny; j++) {
            for (int c = 0; c < pfc->nc; c++) {
                if (zero_vel) 
                    velocity[c][i*pfc->ny + j] = dz * pfc->grad_theta[c][i*pfc->ny + j];
                else
                    velocity[c][i*pfc->ny + j] = gamma*velocity[c][i*pfc->ny+j]
                                            + dz*pfc->grad_theta[c][i*pfc->ny + j];
                pfc->eta[c][i*pfc->ny + j] *= exp(complex<double>(0.0, 1.0)
                        *(-1.0)*velocity[c][i*pfc->ny + j]);
            }
        }
    }
}


/*! 
 *  num_fft is an output parameter, which will contain the total number of ffts taken
 */
int MechanicalEquilibrium::accelerated_steepest_descent_adaptive_dz(int *p_n_fft) {
    double dz_accd = 1.0;
    int max_iter = 10000;
    double tolerance = 7.5e-9;
    //double tolerance = 1.0e-8;
    bool print = false;

    int adaptive_step_freq = 100;
    int num_adaptive_steps = 5;
    int check_freq = 50;

    double gamma = 0.9;
    
    Time::time_point time_start = Time::now();
    Time::time_point time_var = time_start;

    *p_n_fft = 0;

    // Allocate memory to hold velocity values (no need for FFT plans)
    // Note that the actual steps will be taken in negative direction of velocity
    double **velocity= (double **) malloc(sizeof(double*)*pfc->nc);
    for (int i = 0; i < pfc->nc; i++)
        velocity[i] = (double*) malloc(sizeof(double)*pfc->local_nx*pfc->ny);

    // Boolean when to ignore velocity (first iteration and after adaptive steps)
    bool zero_velocity = true;

    double last_energy = pfc->calculate_energy(pfc->eta, pfc->eta_k); (*p_n_fft)++;
    // update eta_k 
    pfc->take_fft(pfc->eta_plan_f); (*p_n_fft)++;

    int it = 1;
    while (it <= max_iter) {
        if ((it-1) % adaptive_step_freq == 0) {
            // -----------------------------------------------
            // Run the specified number of adaptive steps
            for (int sn = 0; sn < num_adaptive_steps; sn++) {
                pfc->calculate_grad_theta(pfc->eta, pfc->eta_k); (*p_n_fft)++;

                double energy_io = pfc->calculate_energy(pfc->eta, pfc->eta_k); (*p_n_fft)++;
                double dz = exp_line_search(&energy_io, pfc->grad_theta, p_n_fft);
                
                double error = elementwise_avg_norm();

                if (pfc->mpi_rank == 0 && print)
                    printf("it: %5d; adaptive step: %6.1f; energy: %.16e; err: %.16e; "
                           "num_fft: %d\n",
                            it, dz, energy_io, error, *p_n_fft);

                last_energy = energy_io;
                it++;
            }
            zero_velocity = true;
            // -----------------------------------------------
        }
        // Resume with accelerated descent
        if (zero_velocity) {
            // If last step velocity is zero (or uninitialized) take normal gradient
            pfc->calculate_grad_theta(pfc->eta, pfc->eta_k); (*p_n_fft)++;

        } else {
            // If last step velocity is not zero, take a prediction gradient
            take_step(gamma, velocity, pfc->eta, pfc->eta_tmp);
            // update eta_tmp_k
            pfc->take_fft(pfc->eta_tmp_plan_f); (*p_n_fft)++;
            // calculate gradient based on eta_tmp_k
            pfc->calculate_grad_theta(pfc->eta_tmp, pfc->eta_tmp_k); (*p_n_fft)++;
        }
        update_velocity_and_take_step(dz_accd, gamma, velocity, zero_velocity);
        pfc->take_fft(pfc->eta_plan_f); (*p_n_fft)++;
        zero_velocity = false;

        if (it % check_freq == 0) {
            double energy = pfc->calculate_energy(pfc->eta, pfc->eta_k); (*p_n_fft)++;
            double error = elementwise_avg_norm();
            if (pfc->mpi_rank == 0 && print) {
                // timings ---------
                double it_dur = std::chrono::duration<double>(Time::now()-time_var).count();
                double tot_dur = std::chrono::
                    duration<double>(Time::now()-time_start).count();
                time_var = Time::now();
                // -----------------
                printf("it: %5d; energy: %.16e; err: %.16e; time: %4.1f; tot_time: %6.1f; "
                       "num_fft: %d\n",
                        it, energy, error, it_dur, tot_dur, *p_n_fft);
                if (energy > last_energy) cout << "Warning: energy increased." << endl;
                if (error < tolerance) cout << "Solution found." << endl;
            }
            last_energy = energy;
            if (error < tolerance) break;
        }
        if (it >= max_iter && print)
            printf("Solution was not found within %d iterations.\n", max_iter);
        it++;
    }

    for (int i = 0; i < pfc->nc; i++)
        fftw_free(velocity[i]);
    free(velocity);

    return it;
}

