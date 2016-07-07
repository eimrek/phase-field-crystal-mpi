
#include <iostream>
#include <stdlib.h>

#include <mpi.h>
#include <fftw3-mpi.h>

#include "pfc.h"

// ---------------------------------------------------------------
// PARAMETERS
//
// initialization could also be done in header, if const->constexpr 

const int PhaseField::nx = 8;
const int PhaseField::ny = 8;

const double PhaseField::dx = 2.0;
const double PhaseField::dy = 2.0;
const double PhaseField::dt = 0.125;

const double PhaseField::q_vec[][2] = 
    {{-0.5*sq3, -0.5},
     {0.0, 1.0},
     {0.5*sq3, -0.5}};

const double PhaseField::bx = 1.0;
const double PhaseField::bl = 0.95;
const double PhaseField::tt = 0.585;
const double PhaseField::vv = 1.0;

const int PhaseField::nc = 3;

// ---------------------------------------------------------------

PhaseField::PhaseField(int mpi_size, int mpi_rank)
        : mpi_size(mpi_size), mpi_rank(mpi_rank) {

    fftw_mpi_init(); 
   
    // Allocate and calculate k values
    k_x_values = (double*) malloc(sizeof(double)*nx);
    k_y_values = (double*) malloc(sizeof(double)*ny);
    calculate_k_values(k_x_values, nx, dx);
    calculate_k_values(k_y_values, ny, dy);
    
    // Allocate etas and plans FFT plans and same for buffer values
    eta = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    eta_k = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    plan_forward = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);
    plan_backward = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);

    buffer = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    buffer_k = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    buffer_plan_f = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);
    buffer_plan_b = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);


    ptrdiff_t alloc_local = fftw_mpi_local_size_2d(nx, ny, MPI_COMM_WORLD,
            &local_nx, &local_nx_start);

    // Allocate and calculate G_j values
    g_values = (double**) malloc(sizeof(double*)*nc);
    for (int i = 0; i < nc; i++) {
        g_values[i] = (double*) malloc(sizeof(double)*local_nx*ny);
    }
    calculate_g_values(g_values);


    for (int i = 0; i < nc; i++) {
        eta[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
        eta_k[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
        plan_forward[i] = fftw_mpi_plan_dft_2d(nx, ny,
                reinterpret_cast<fftw_complex*>(eta[i]),
                reinterpret_cast<fftw_complex*>(eta_k[i]),
                MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward[i] = fftw_mpi_plan_dft_2d(nx, ny,
                reinterpret_cast<fftw_complex*>(eta_k[i]),
                reinterpret_cast<fftw_complex*>(eta[i]),
                MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

        buffer[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
        buffer_k[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
        buffer_plan_f[i] = fftw_mpi_plan_dft_2d(nx, ny,
                reinterpret_cast<fftw_complex*>(buffer[i]),
                reinterpret_cast<fftw_complex*>(buffer_k[i]),
                MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
        buffer_plan_b[i] = fftw_mpi_plan_dft_2d(nx, ny,
                reinterpret_cast<fftw_complex*>(buffer_k[i]),
                reinterpret_cast<fftw_complex*>(buffer[i]),
                MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

}

PhaseField::~PhaseField() {
    for (int i = 0; i < nc; i++) {
        fftw_free(eta[i]); fftw_free(eta_k[i]);
        fftw_destroy_plan(plan_forward[i]); fftw_destroy_plan(plan_backward[i]); 

        fftw_free(buffer[i]); fftw_free(buffer_k[i]);
        fftw_destroy_plan(buffer_plan_f[i]); fftw_destroy_plan(buffer_plan_b[i]); 

        free(g_values[i]);
    }
    free(eta); free(eta_k);
    free(plan_forward); free(plan_backward);

    free(buffer); free(buffer_k);
    free(buffer_plan_f); free(buffer_plan_b);

    free(k_x_values); free(k_y_values);
    free(g_values);
}

/*! Method, that initializes the state to a elastically rotated circle
 *
 */
void PhaseField::initialize_eta() {
    double angle = 0.0872665;
    double amplitude = 0.10867304595992146;

    for (int i = 0; i < local_nx; i++) {
        int i_gl = i + local_nx_start;
        for (int j = 0; j < ny; j++) {
            for (int c = 0; c < nc; c++) {
                if ((i_gl+1-nx/2.0)*(i_gl+1-nx/2.0)*dx*dx + (j+1-ny/2.0)*(j+1-ny/2.0)*dy*dy
                        <= (0.25*nx*dx)*(0.25*nx*dx)) {
                    // Inside the rotated circle
                    double theta = (q_vec[c][0]*cos(angle) + q_vec[c][1]*sin(angle)
                                    - q_vec[c][0])*(i_gl+1-nx/2.0)*dx
                                 + (-q_vec[c][0]*sin(angle) + q_vec[c][1]*cos(angle)
                                    - q_vec[c][1])*(j+1-ny/2.0)*dy;
                    eta[c][i*ny + j] = amplitude*exp(J*theta);
                } else {
                    // Outside the circle
                    eta[c][i*ny + j] = amplitude;
                }
            }
        }
    }
}

void PhaseField::take_fft(fftw_plan *plan) {
    for (int i = 0; i < nc; i++) {
        fftw_execute(plan[i]);
    }
}

void PhaseField::normalize_field(complex<double> **field) {
    double scale = 1.0/(nx*ny);
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int n = 0; n < nc; n++) {
                field[n][i*ny + j] *= scale;
            }
        }
    }
}

complex<double>* PhaseField::get_eta(int num) {
    return eta[num];
}

complex<double>* PhaseField::get_eta_k(int num) {
    return eta_k[num];
}

/*! Method that gathers all eta or eta_k data to root process and prints it out
 *  
 *  NB: The whole data must fit inside root process memory
 */
void PhaseField::output_field(complex<double>*field) {
    complex<double>* field_total;
    if (mpi_rank == 0) {
        field_total = (complex<double>*) fftw_malloc(sizeof(complex<double>)*nx*ny);
    }
    // local_nx*ny*2, because one fftw_complex element contains 2 doubles
    MPI_Gather(field, local_nx*ny*2, MPI_DOUBLE, field_total, local_nx*ny*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        for (int i = 0; i < nx; i++) {
            std::cout << "|";
            for (int j = 0; j < ny; j++) {
                printf("%11.4e ", real(field_total[i*ny+j]));
            }
            std::cout << "| |";
    
            for (int j = 0; j < ny; j++) {
                printf("%11.4e ", imag(field_total[i*ny+j]));
            }
            std::cout << "|" << std::endl;
        }
    
        fftw_free(field_total);
        std::cout << std::endl;
    }
}

/*! Method to calculate the wave number values corresponding to bins in k space
 *
 *  The method was written on basis of numpy.fft.fftfreq() 
 */
void PhaseField::calculate_k_values(double *k_values, int n, double d) {
    for (int i = 0; i < n; i++) {
        if (n % 2 == 0) {
        // If n is even
            if (i < n/2) k_values[i] = 2*PI*i/(n*d);
            else k_values[i] = 2*PI*(i-n)/(n*d);
        } else {
        // Else n is odd
            if (i <= n/2) k_values[i] = 2*PI*i/(n*d);
            else k_values[i] = 2*PI*(i-n)/(n*d);
        }
    }
}

/*! Method that calculates G_j values in k space
 *  
 */
void PhaseField::calculate_g_values(double **g_values) {
    for (int i = 0; i < local_nx; i++) {
        int i_shift = i + local_nx_start;
        for (int j = 0; j < ny; j++) {
            double k_sq = k_x_values[i_shift]*k_x_values[i_shift]
                + k_y_values[j]*k_y_values[j];
            for (int n = 0; n < 3; n++) {
                g_values[n][i*ny + j] = - k_sq - 2*(q_vec[n][0]*k_x_values[i_shift]
                        + q_vec[n][1]*k_y_values[j]);
            }
        }
    }
}

/*! Method to calculate energy.
 *
 *  NB: This method assumes that eta_k is set beforehand.
 */
double PhaseField::calculate_energy() {
    // will use the member variable buffer_k to hold (G_j eta_j)_k
    for (int c = 0; c < nc; c++) {
        memcpy(buffer_k[c], eta_k[c], sizeof(complex<double>)*local_nx*ny);
    }

    //  Multiply eta_k by G_j in k space
    for (int c = 0; c < nc; c++) {
        for (int i = 0; i < local_nx; i++) {
            for (int j = 0; j < ny; j++) {
                buffer_k[c][i*ny + j] *= g_values[c][i*ny + j];
            }
        }
    }

    // Go to real space for (G_j eta_j)
    take_fft(buffer_plan_b);
    normalize_field(buffer);


    // Integrate the whole expression over space and divide by num cells to get density
    // NB: this will be the contribution from local MPI process only
    double local_energy = 0.0;
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            complex<double> eta0 = eta[0][i*ny+j]; complex<double> buf0 = buffer[0][i*ny+j];
            complex<double> eta1 = eta[1][i*ny+j]; complex<double> buf1 = buffer[1][i*ny+j];
            complex<double> eta2 = eta[2][i*ny+j]; complex<double> buf2 = buffer[2][i*ny+j];
            double aa = 2*(abs(eta0)*abs(eta0) + abs(eta1)*abs(eta1) + abs(eta2)*abs(eta2));

            local_energy += aa*(bl-bx)/2.0 + (3.0/4.0)*vv*aa*aa
                         - 4*tt*real(eta0*eta1*eta2)
                         + bx*(abs(buf0)*abs(buf0)+abs(buf1)*abs(buf1)+abs(buf2)*abs(buf2))
                         - (3.0/2.0)*vv*(abs(eta0)*abs(eta0)*abs(eta0)*abs(eta0)
                                            + abs(eta1)*abs(eta1)*abs(eta1)*abs(eta1)
                                            + abs(eta2)*abs(eta2)*abs(eta2)*abs(eta2));
        }
    }
    local_energy *= 1.0/(nx*ny);

    double energy = 0.0;
    
    MPI_Allreduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("Energy: %.16e\n", energy);
    }
    return energy;
}

/*! Method, that calculates the three components of the nonlinear part
 *
 *  The components will be saved to components (memory must be allocated before)
 *  and they correspond to real space indices (coordinates) of i, j
 */
void PhaseField::calculate_nonlinear_part(int i, int j, complex<double> *components) {
    complex<double> eta0 = eta[0][i*ny+j];
    complex<double> eta1 = eta[1][i*ny+j];
    complex<double> eta2 = eta[2][i*ny+j];
    double aa = 2*(abs(eta0)*abs(eta0) + abs(eta1)*abs(eta1) + abs(eta2)*abs(eta2));
    components[0] = 3*vv*(aa-abs(eta0)*abs(eta0))*eta0 - 2*tt*conj(eta1)*conj(eta2);
    components[1] = 3*vv*(aa-abs(eta1)*abs(eta1))*eta1 - 2*tt*conj(eta0)*conj(eta2);
    components[2] = 3*vv*(aa-abs(eta2)*abs(eta2))*eta2 - 2*tt*conj(eta1)*conj(eta0);
}

/*! Method, which takes an overdamped dynamics time step
 *
 */
void PhaseField::overdamped_time_step() {
    // Will use buffer to hold intermediate results
    for (int c = 0; c < nc; c++) {
        memcpy(buffer[c], eta[c], sizeof(complex<double>)*local_nx*ny);
    }
    // allocate memory for the nonlinear part
    complex<double> *nonlinear_part = (complex<double>*) malloc(sizeof(complex<double>)*nc);

    // buffer contains eta, let's subtract dt*(nonlinear part)
    // to get the numerator in the OD time stepping scheme (in real space)
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            calculate_nonlinear_part(i, j, nonlinear_part);
            for (int c = 0; c < nc; c++) {
                buffer[c][i*ny + j] -= dt*nonlinear_part[c];
            }
        }
    }
    free(nonlinear_part);
    
    // take buffer into k space
    take_fft(buffer_plan_f);
    
    // now eta_k can be evaluated correspondingly to the scheme
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int c = 0; c < nc; c++) {
                eta_k[c][i*ny + j] = buffer_k[c][i*ny + j] / (1.0 + dt*(bl-bx
                                        +bx*g_values[c][i*ny + j]*g_values[c][i*ny + j]));
            }
        }
    }

    // Take eta back to real space
    take_fft(plan_backward);
    normalize_field(eta);

}

double PhaseField::dot_prod(const double* v1, const double* v2, const int len) {
    double res = 0.0;
    for (int i = 0; i < len; i++) {
        res += v1[i]*v2[i];
    }
    return res;
}

/*! Method, which calculates the gradient of thetas
 *
 *  The resulting gradient will be stored in "buffer"
 */
void PhaseField::calculate_grad_theta() {
    // will use the member variable buffer_k to hold (G_j^2 eta_j)_k
    for (int c = 0; c < nc; c++) {
        memcpy(buffer_k[c], eta_k[c], sizeof(complex<double>)*local_nx*ny);
    }

    //  Multiply eta_k by G_j^2 in k space
    for (int c = 0; c < nc; c++) {
        for (int i = 0; i < local_nx; i++) {
            for (int j = 0; j < ny; j++) {
                buffer_k[c][i*ny + j] *= g_values[c][i*ny + j]*g_values[c][i*ny + j];
            }
        }
    }

    // Go to real space for (G_j^2 eta_j)
    take_fft(buffer_plan_b);
    normalize_field(buffer);
    

    // allocate memory for the nonlinear part and a temporary result
    complex<double> *nonlinear_part = (complex<double>*) malloc(sizeof(complex<double>)*nc);
    double *im = (double*) malloc(sizeof(double)*nc);

    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
           calculate_nonlinear_part(i, j, nonlinear_part);
           for (int c = 0; c < nc; c++) {
                complex<double> var_f_eta = (bl-bx)*eta[c][i*ny+j] + bx*buffer[c][i*ny+j]
                                            + nonlinear_part[c];
                //if (c == 0) printf("%.3e\n", var_f_eta);
                im[c] = imag(conj(eta[c][i*ny+j])*var_f_eta);
                //if (c == 0) printf("%.3e %.3e %.3e\n", im[c], var_f_eta, eta[c][i*ny+j]);
           }
           for (int c = 0; c < nc; c++) {
                double dtheta = dot_prod(q_vec[c], q_vec[0], nc)*im[0]
                               + dot_prod(q_vec[c], q_vec[1], nc)*im[1]
                               + dot_prod(q_vec[c], q_vec[2], nc)*im[2];
                buffer[c][i*ny+j] = dtheta;
           }
        }
    }
    
    free(nonlinear_part);
    free(im);
}

void PhaseField::test() {
    //output_field(eta[0]);

    //take_fft(plan_forward);
    //calculate_energy();
    //overdamped_time_step();
    //calculate_energy();
    //output_field(eta[0]);
    //calculate_grad_theta();
    //output_field(buffer[0]);
    cout << "test ran" << endl;
}

