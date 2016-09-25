
#include <iostream>
#include <cstring>
#include <cstdlib>

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
    for (int c = 0; c < pfc->nc; c++) {
		for (int i = 0; i < pfc->local_nx; i++) {
			for (int j = 0; j < pfc->ny; j++) {
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
    for (int c = 0; c < pfc->nc; c++) {
		for (int i = 0; i < pfc->local_nx; i++) {
			for (int j = 0; j < pfc->ny; j++) {
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
 *  Finds first suitable step by trying exponentially increasing steps
 *  Will also set "eta" and "eta_k"
 *
 *  NB: Might be slow due to a lot of memory copying...
 *  (Or it might be negligible compared to ffts)
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
    pfc->take_fft(pfc->eta_tmp_plan_f);
    double energy = pfc->calculate_energy(pfc->eta_tmp, pfc->eta_tmp_k);

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
        pfc->take_fft(pfc->eta_tmp_plan_f);
        double energy = pfc->calculate_energy(pfc->eta_tmp, pfc->eta_tmp_k);

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

int MechanicalEquilibrium::steepest_descent_line_search() {
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
        double dz = exp_line_search(&energy, pfc->grad_theta);

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


int MechanicalEquilibrium::accelerated_gradient_descent(
		double dz, int max_iter, double tolerance, bool print
		) {
	//double dz = 1.0;
	//int max_iter = 10000;
	//double tolerance = 1.0e-7;
	//bool print = true;

	int check_freq = 100;

	double gamma = 0.992;

	//Time::time_point time_start = Time::now();
	Time::time_point time_var = Time::now();

	// Allocate memory to hold velocity values (no need for FFT plans)
	// Note that the actual steps will be taken in negative direction of velocity
	double **velocity= (double **) malloc(sizeof(double*)*pfc->nc);
	for (int i = 0; i < pfc->nc; i++)
		velocity[i] = (double*) malloc(sizeof(double)*pfc->local_nx*pfc->ny);

	double last_energy = pfc->calculate_energy(pfc->eta, pfc->eta_k);
	// update eta_k
	pfc->take_fft(pfc->eta_plan_f);

	int it = 1;
	while (it <= max_iter) {
		if (it == 1) {
			// If last step velocity is zero (or uninitialized) take normal gradient
			pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);

		} else {
			// If last step velocity is not zero, take a prediction gradient
			take_step(gamma, velocity, pfc->eta, pfc->eta_tmp);
			// update eta_tmp_k
			pfc->take_fft(pfc->eta_tmp_plan_f);
			// calculate gradient based on eta_tmp
			pfc->calculate_grad_theta(pfc->eta_tmp, pfc->eta_tmp_k);
		}
		update_velocity_and_take_step(dz, gamma, velocity, it == 1);

		if (it % check_freq == 0) {
			pfc->take_fft(pfc->eta_plan_f);
			double energy = pfc->calculate_energy(pfc->eta, pfc->eta_k);
			double error = elementwise_avg_norm();
			if (pfc->mpi_rank == 0 && print) {
				// timings ---------
				double it_dur = std::chrono::duration<double>(Time::now()-time_var).count();
				//double tot_dur = std::chrono::
				//	duration<double>(Time::now()-time_start).count();
				time_var = Time::now();
				// -----------------
				printf("    it: %5d; energy: %.16e; err: %.16e; time: %3.1f\n",
						it, energy, error, it_dur);
				if (energy > last_energy) cout << "    Warning: energy increased." << endl;
				if (error < tolerance) cout << "    Solution found." << endl;
			}
			last_energy = energy;
			if (error < tolerance) break;
		}
		if (it >= max_iter && print && pfc->mpi_rank == 0)
			printf("    Solution was not found within %d iterations.\n", max_iter);
		it++;
	}

	for (int i = 0; i < pfc->nc; i++)
		free(velocity[i]);
	free(velocity);

	return it;

}


void MechanicalEquilibrium::update_velocity_and_take_step(double dz, double gamma,
        double **velocity, bool zero_vel) {
    for (int c = 0; c < pfc->nc; c++) {
		for (int i = 0; i < pfc->local_nx; i++) {
			for (int j = 0; j < pfc->ny; j++) {
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
 *  Accelerated steepest descent with occasional line search
 */
int MechanicalEquilibrium::accelerated_gradient_descent_line_search() {
    double dz_accd = 1.0;
    int max_iter = 10000;
    double tolerance = 7.5e-9;
    //double tolerance = 1.0e-8;
    bool print = true;

    int adaptive_step_freq = 100;
    int num_adaptive_steps = 5;
    int check_freq = 50;

    double gamma = 0.9;
    
    Time::time_point time_start = Time::now();
    Time::time_point time_var = time_start;


    // Allocate memory to hold velocity values (no need for FFT plans)
    // Note that the actual steps will be taken in negative direction of velocity
    double **velocity= (double **) malloc(sizeof(double*)*pfc->nc);
    for (int i = 0; i < pfc->nc; i++)
        velocity[i] = (double*) malloc(sizeof(double)*pfc->local_nx*pfc->ny);

    // Boolean when to ignore velocity (first iteration and after adaptive steps)
    bool zero_velocity = true;

    double last_energy = pfc->calculate_energy(pfc->eta, pfc->eta_k);
    // update eta_k 
    pfc->take_fft(pfc->eta_plan_f);

    int it = 1;
    while (it <= max_iter) {
        if ((it-1) % adaptive_step_freq == 0) {
            // -----------------------------------------------
            // Run the specified number of adaptive steps
            for (int sn = 0; sn < num_adaptive_steps; sn++) {
                pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);

                double energy_io = pfc->calculate_energy(pfc->eta, pfc->eta_k);
                double dz = exp_line_search(&energy_io, pfc->grad_theta);
                
                double error = elementwise_avg_norm();

                if (pfc->mpi_rank == 0 && print)
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
            pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);

        } else {
            // If last step velocity is not zero, take a prediction gradient
            take_step(gamma, velocity, pfc->eta, pfc->eta_tmp);
            // update eta_tmp_k
            pfc->take_fft(pfc->eta_tmp_plan_f);
            // calculate gradient based on eta_tmp_k
            pfc->calculate_grad_theta(pfc->eta_tmp, pfc->eta_tmp_k);
        }
        update_velocity_and_take_step(dz_accd, gamma, velocity, zero_velocity);
        zero_velocity = false;

        if (it % check_freq == 0) {
            pfc->take_fft(pfc->eta_plan_f);
            double energy = pfc->calculate_energy(pfc->eta, pfc->eta_k);
            double error = elementwise_avg_norm();
            if (pfc->mpi_rank == 0 && print) {
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
        if (it >= max_iter && print)
            printf("Solution was not found within %d iterations.\n", max_iter);
        it++;
    }

    for (int i = 0; i < pfc->nc; i++)
        free(velocity[i]);
    free(velocity);

    return it;
}

double MechanicalEquilibrium::dot_prod(double **v1, double **v2) {
	double res = 0.0;
	for (int c = 0; c < pfc->nc; c++)
		for (int i = 0; i < pfc->local_nx; i++)
			for (int j = 0; j < pfc->ny; j++)
				res += v1[c][i*pfc->ny+j] * v2[c][i*pfc->ny+j];
	MPI_Allreduce(&res, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return res;
}

void MechanicalEquilibrium::lbfgs_direction(int m, double ***s, double ***y,
		double **grad, double **result) {

	double* alpha = (double*) malloc(sizeof(double)*m);
	double* rho = (double*) malloc(sizeof(double)*m);

	// Copy gradient to result
	for (int c = 0; c < pfc->nc; c++)
		std::memcpy(result[c], grad[c], sizeof(double)*pfc->local_nx*pfc->ny);

	for (int i_m = 0; i_m < m; i_m++) {

		rho[i_m] = 1.0/dot_prod(s[i_m], y[i_m]);

		alpha[i_m] = rho[i_m] * dot_prod(s[i_m], result);

		for (int c = 0; c < pfc->nc; c++)
			for (int i = 0; i < pfc->local_nx; i++)
				for (int j = 0; j < pfc->ny; j++)
					result[c][i*pfc->ny+j] -= alpha[i_m] * y[i_m][c][i*pfc->ny+j];
	}
	// H_0 = Identity matrix, so "r = q"

	for (int i_m = m-1; i_m > -1; i_m--) {
		double beta = rho[i_m] * dot_prod(y[i_m], result);

		for (int c = 0; c < pfc->nc; c++)
			for (int i = 0; i < pfc->local_nx; i++)
				for (int j = 0; j < pfc->ny; j++)
					result[c][i*pfc->ny+j] += s[i_m][c][i*pfc->ny+j]*(alpha[i_m]-beta);
	}

	free(alpha);
	free(rho);
}

/* Moves queue such that first element points to second and so on..
 * final element will point to the first (and the memory can be changed)
 */
void MechanicalEquilibrium::move_queue(int m, double ***queue) {
	double **temp = queue[0];
	for (int i = 0; i < m-1; i++) {
		queue[i] = queue[i+1];
	}
	queue[m-1] = temp;
}

/* NB! This method is fairly sensitive to numerical noise;
 * if FFTW_MEASURE (instead of FFTW_ESTIMATE) is used for the fftw plan,
 * different runs on same machine will yield different results
 * (number of iterations might fluctuate by ~1000)
 *
 */
int MechanicalEquilibrium::lbfgs() {

	double tolerance = 7.5e-9;
	//double tolerance = 1.0e-7;
	int check_freq = 100;
	bool print = true;

	int m = 5;
	double dz = 1.0;

	Time::time_point time_var = Time::now();

	// -----------------------------------------------------------------------------
	// Memory allocations
	// Will hold the arrays to theta and grad differences for past states
	double*** s = (double***) malloc(sizeof(double**)*m);
	double*** y = (double***) malloc(sizeof(double**)*m);
	for (int i = 0; i < m; i++) {
		s[i] = (double**) malloc(sizeof(double*)*pfc->nc);
		y[i] = (double**) malloc(sizeof(double*)*pfc->nc);
		for (int c = 0; c < pfc->nc; c++) {
			s[i][c] = (double*) malloc(sizeof(double)*pfc->local_nx*pfc->ny);
			y[i][c] = (double*) malloc(sizeof(double)*pfc->local_nx*pfc->ny);
		}
	}

	// Direction of the LBFGS step will be saved here
	// Previous step theta and grad are saved here
	double** lbfgs_dir = (double**) malloc(sizeof(double*)*pfc->nc);
	double** prev_grad = (double**) malloc(sizeof(double*)*pfc->nc);
	for (int c = 0; c < pfc->nc; c++) {
		lbfgs_dir[c] = (double*) malloc(sizeof(double)*pfc->local_nx*pfc->ny);
		prev_grad[c] = (double*) malloc(sizeof(double)*pfc->local_nx*pfc->ny);
	}
	// -----------------------------------------------------------------------------
	// Initial gradient
	pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);

	for (int c = 0; c < pfc->nc; c++)
		std::memcpy(prev_grad[c], pfc->grad_theta[c], sizeof(double)*pfc->local_nx*pfc->ny);
	// -----------------------------------------------------------------------------

	int m_c = 0; // current changed value; goes up to m-1
	int m_q = 0; // queue length (s, y); goes up to m

	int max_it = 10000;
	int it = 0;
	for (; it < max_it; it++) {

		lbfgs_direction(m_q, s, y, prev_grad, lbfgs_dir);

		// Move queues if they're "full"
		if (m_q == m) {
			move_queue(m, s);
			move_queue(m, y);
		}

		// take step and update s
		for (int c = 0; c < pfc->nc; c++)
			for (int i = 0; i < pfc->local_nx; i++)
				for (int j = 0; j < pfc->ny; j++) {
					double dtheta = - dz*lbfgs_dir[c][i*pfc->ny+j];
					pfc->eta[c][i*pfc->ny+j] *= std::exp(complex<double>(0.0, 1.0)*dtheta);
					s[m_c][c][i*pfc->ny+j] = dtheta;
				}
		// update eta_k, calculate new gradient and update y
		pfc->take_fft(pfc->eta_plan_f);
		pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);
		for (int c = 0; c < pfc->nc; c++)
			for (int i = 0; i < pfc->local_nx; i++)
				for (int j = 0; j < pfc->ny; j++) {
					y[m_c][c][i*pfc->ny+j] = pfc->grad_theta[c][i*pfc->ny+j]-prev_grad[c][i*pfc->ny+j];
				}

		for (int c = 0; c < pfc->nc; c++)
			std::memcpy(prev_grad[c], pfc->grad_theta[c], sizeof(double)*pfc->local_nx*pfc->ny);

		if (m_c < m-1) m_c++;
		if (m_q < m) m_q++;

		double error = elementwise_avg_norm();
		if (it % check_freq == 0 || error < tolerance) {
			double energy = pfc->calculate_energy(pfc->eta, pfc->eta_k);
			double dur = std::chrono::duration<double>(Time::now()-time_var).count();
			time_var = Time::now();
			if (pfc->mpi_rank == 0 && print) {
				printf("it: %5d; energy: %.14e; err: %.14e; time: %.1f\n",
						it, energy, error, dur);
			}
			if (error < tolerance)
				break;
		}
	}

	// -------------------------------------------------------------------
	// Free memory
	for (int i = 0; i < m; i++) {
		for (int c = 0; c < pfc->nc; c++) {
			free(s[i][c]);
			free(y[i][c]);
		}
		free(s[i]);
		free(y[i]);
	}
	free(s);
	free(y);
	for (int c = 0; c < pfc->nc; c++) {
		free(lbfgs_dir[c]);
		free(prev_grad[c]);
	}
	free(lbfgs_dir);
	free(prev_grad);
	// -------------------------------------------------------------------

	return it;
}



int MechanicalEquilibrium::lbfgs_iterations = 500;

int MechanicalEquilibrium::lbfgs_enhanced() {

	//double tolerance = 7.5e-9;
	double tolerance = 1.0e-8;
	int check_freq = 100;
	bool print = true;

	int m = 5;
	double dz = 1.0;

	int accelerated_descent_iterations = 400;
	int lbfgs_it_increase = 500;

	Time::time_point time_var = Time::now();

	// -----------------------------------------------------------------------------
	// Memory allocations
	// Will hold the arrays to theta and grad differences for past states
	double*** s = (double***) malloc(sizeof(double**)*m);
	double*** y = (double***) malloc(sizeof(double**)*m);
	for (int i = 0; i < m; i++) {
		s[i] = (double**) malloc(sizeof(double*)*pfc->nc);
		y[i] = (double**) malloc(sizeof(double*)*pfc->nc);
		for (int c = 0; c < pfc->nc; c++) {
			s[i][c] = (double*) malloc(sizeof(double)*pfc->local_nx*pfc->ny);
			y[i][c] = (double*) malloc(sizeof(double)*pfc->local_nx*pfc->ny);
		}
	}

	// Direction of the LBFGS step will be saved here
	// Previous step theta and grad are saved here
	double** lbfgs_dir = (double**) malloc(sizeof(double*)*pfc->nc);
	double** prev_grad = (double**) malloc(sizeof(double*)*pfc->nc);
	for (int c = 0; c < pfc->nc; c++) {
		lbfgs_dir[c] = (double*) malloc(sizeof(double)*pfc->local_nx*pfc->ny);
		prev_grad[c] = (double*) malloc(sizeof(double)*pfc->local_nx*pfc->ny);
	}
	// -----------------------------------------------------------------------------

	int total_lbfgs_iterations = 0;
	double error = 1.0;

	int rep = 0;
	for (; rep < 100; rep++) {

		// -----------------------------------------------------------------------------------
		// 1) Line search
		// Update gradient and calc energy
		pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);
		double last_energy = pfc->calculate_energy(pfc->eta, pfc->eta_k);

		// Do the exponential line search to find optimal step
		// will update eta, eta_k and also store new energy value
		double dz_ls = exp_line_search(&last_energy, pfc->grad_theta);

		if (pfc->mpi_rank == 0 && print) printf("    Line search step: %.2f\n", dz_ls);
		// -----------------------------------------------------------------------------------
		// 2) LBFGS steps

		// update and store gradient to prev_grad
		pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);
		for (int c = 0; c < pfc->nc; c++)
			std::memcpy(prev_grad[c], pfc->grad_theta[c], sizeof(double)*pfc->local_nx*pfc->ny);

		int m_c = 0; // current changed value; goes up to m-1
		int m_q = 0; // queue length (s, y); goes up to m

		double current_lbfgs_iterations = 0;
		if (rep == 0) current_lbfgs_iterations = lbfgs_iterations;
		else current_lbfgs_iterations = lbfgs_it_increase;

		for (int it = 1; it < current_lbfgs_iterations + 1; it++) {
			lbfgs_direction(m_q, s, y, prev_grad, lbfgs_dir);

			// Move queues if they're "full"
			if (m_q == m) {
				move_queue(m, s);
				move_queue(m, y);
			}

			// take step and update s
			for (int c = 0; c < pfc->nc; c++)
				for (int i = 0; i < pfc->local_nx; i++)
					for (int j = 0; j < pfc->ny; j++) {
						double dtheta = - dz*lbfgs_dir[c][i*pfc->ny+j];
						pfc->eta[c][i*pfc->ny+j] *= std::exp(complex<double>(0.0, 1.0)*dtheta);
						s[m_c][c][i*pfc->ny+j] = dtheta;
					}
			// update eta_k, calculate new gradient and update y
			pfc->take_fft(pfc->eta_plan_f);
			pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);
			for (int c = 0; c < pfc->nc; c++)
				for (int i = 0; i < pfc->local_nx; i++)
					for (int j = 0; j < pfc->ny; j++) {
						y[m_c][c][i*pfc->ny+j] = pfc->grad_theta[c][i*pfc->ny+j]-prev_grad[c][i*pfc->ny+j];
					}

			for (int c = 0; c < pfc->nc; c++)
				std::memcpy(prev_grad[c], pfc->grad_theta[c], sizeof(double)*pfc->local_nx*pfc->ny);

			if (m_c < m-1) m_c++;
			if (m_q < m) m_q++;

			total_lbfgs_iterations++;

			error = elementwise_avg_norm();
			if (it % check_freq == 0 || error < tolerance) {
				double energy = pfc->calculate_energy(pfc->eta, pfc->eta_k);
				double dur = std::chrono::duration<double>(Time::now()-time_var).count();
				time_var = Time::now();
				if (pfc->mpi_rank == 0 && print) {
					printf("    it: %5d; energy: %.14e; err: %.14e; time: %.1f\n",
							it, energy, error, dur);
				}
				if (pfc->mpi_rank == 0 && energy > last_energy)
					printf("    Warning: energy increased during LBFGS steps!\n");
				last_energy = energy;
				if (error < tolerance) break;
			}
		}
		if (error < tolerance) break;
		// -----------------------------------------------------------------------------------
		// 3) Error reducing accelerated descent
		if (pfc->mpi_rank == 0 && print) printf("    Error reduction:\n");
		accelerated_gradient_descent(dz, accelerated_descent_iterations, tolerance, print);

		pfc->calculate_grad_theta(pfc->eta, pfc->eta_k);
		error = elementwise_avg_norm();
		if (error < tolerance) break;
	}

	lbfgs_iterations = total_lbfgs_iterations;

	// -------------------------------------------------------------------
	// Free memory
	for (int i = 0; i < m; i++) {
		for (int c = 0; c < pfc->nc; c++) {
			free(s[i][c]);
			free(y[i][c]);
		}
		free(s[i]);
		free(y[i]);
	}
	free(s);
	free(y);
	for (int c = 0; c < pfc->nc; c++) {
		free(lbfgs_dir[c]);
		free(prev_grad[c]);
	}
	free(lbfgs_dir);
	free(prev_grad);
	// -------------------------------------------------------------------

	return rep;
}


