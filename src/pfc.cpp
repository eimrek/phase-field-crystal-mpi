
#include <iostream>
#include <stdlib.h>

#include <mpi.h>
#include <fftw3-mpi.h>

#include "pfc.h"

const double PhaseField::q_vectors[][2] = 
    {{-0.5*sq3, -0.5},
     {0.0, 1.0},
     {0.5*sq3, -0.5}};

PhaseField::PhaseField(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    fftw_mpi_init(); 
   
    // Allocate and calculate k values
    k_x_values = (double*) malloc(sizeof(double)*nx);
    k_y_values = (double*) malloc(sizeof(double)*ny);
    calculate_k_values(k_x_values, nx, dx);
    calculate_k_values(k_y_values, ny, dy);
    
    // Allocate and calculate G_j values
    g_values = (double**) malloc(sizeof(double*)*nc);
    for (int i = 0; i < nc; i++) {
        g_values[i] = (double*) malloc(sizeof(double)*nx*ny);
    }
    calculate_g_values(g_values);

    // Allocate etas and plans FFT plans and same for buffer values
    eta = (fftw_complex**) malloc(sizeof(fftw_complex*)*nc);
    eta_k = (fftw_complex**) malloc(sizeof(fftw_complex*)*nc);
    plan_forward = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);
    plan_backward = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);

    buffer = (fftw_complex**) malloc(sizeof(fftw_complex*)*nc);
    buffer_k = (fftw_complex**) malloc(sizeof(fftw_complex*)*nc);
    buffer_plan_f = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);
    buffer_plan_b = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);


    alloc_local = fftw_mpi_local_size_2d(nx, ny, MPI_COMM_WORLD,
            &local_nx, &local_nx_start);

    for (int i = 0; i < nc; i++) {
        eta[i] = fftw_alloc_complex(alloc_local);
        eta_k[i] = fftw_alloc_complex(alloc_local);
        plan_forward[i] = fftw_mpi_plan_dft_2d(nx, ny, eta[i], eta_k[i], MPI_COMM_WORLD,
            FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward[i] = fftw_mpi_plan_dft_2d(nx, ny, eta_k[i], eta[i], MPI_COMM_WORLD,
            FFTW_BACKWARD, FFTW_ESTIMATE);

        buffer[i] = fftw_alloc_complex(alloc_local);
        buffer_k[i] = fftw_alloc_complex(alloc_local);
        buffer_plan_f[i] = fftw_mpi_plan_dft_2d(nx, ny, buffer[i], 
                buffer_k[i], MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
        buffer_plan_b[i] = fftw_mpi_plan_dft_2d(nx, ny, buffer_k[i],
                buffer[i], MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
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
    MPI_Finalize();
}

void PhaseField::initialize_eta() {
   ptrdiff_t i, j, n;
    for (i = 0; i < local_nx; i++) {
        for (j = 0; j < ny; j++) {
            for (n = 0; n < 3; n++) {
                eta[n][i*ny + j][REAL] = (local_nx_start+i)+j;
                eta[n][i*ny + j][IMAG] = j;
            }
        }
    }
}

void PhaseField::take_fft(fftw_plan *plan) {
    for (int i = 0; i < nc; i++) {
        fftw_execute(plan[i]);
    }
}

void PhaseField::normalize_field(fftw_complex **field) {
    double scale = 1.0/(nx*ny);
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int n = 0; n < nc; n++) {
                field[n][i*ny + j][REAL] *= scale;
                field[n][i*ny + j][IMAG] *= scale;
            }
        }
    }
}

fftw_complex* PhaseField::get_eta(int num) {
    return eta[num];
}

fftw_complex* PhaseField::get_eta_k(int num) {
    return eta_k[num];
}

/*! Method that gathers all eta or eta_k data to root process and prints it out
 *  
 *  NB: The whole data must fit inside root process memory
 */
void PhaseField::output_field(fftw_complex *field) {
    fftw_complex* field_total;
    if (mpi_rank == 0) {
        field_total = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
    }
    // local_nx*ny*2, because one fftw_complex element contains 2 doubles
    MPI_Gather(field, local_nx*ny*2, MPI_DOUBLE, field_total, local_nx*ny*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        for (int i = 0; i < nx; i++) {
            std::cout << "|";
            for (int j = 0; j < ny; j++) {
                printf("%5.1f ", field_total[i*ny+j][REAL]);
            }
            std::cout << "|   |";
    
            for (int j = 0; j < ny; j++) {
                printf("%5.1f ", field_total[i*ny+j][IMAG]);
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

/*! Method that calculated G_j values in k space
 *  
 *  NB: ACTUALLY EACH PROCESS NEEDS ONLY LOCAL DATA, CHANGE IT!
 */
void PhaseField::calculate_g_values(double **g_values) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double k_sq = k_x_values[i]*k_x_values[i] + k_y_values[j]*k_y_values[j];
            for (int n = 0; n < 3; n++) {
                g_values[n][i*ny + j] = - k_sq - 2*(q_vectors[n][0]*k_x_values[i]
                        + q_vectors[n][1]*k_y_values[j]);
            }
        }
    }
}
/*
double PhaseField::abs_squared(fftw_complex *field) {
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            1; 
        }
    }
}
*/

/*! Method to calculate energy.
 *
 *  NB: This method assumes that eta_k is set beforehand.
 */
double PhaseField::calculate_energy() {
    // will use the member variable buffer_k to hold (G_j eta_j)_k
    for (int c = 0; c < nc; c++) {
        memcpy(buffer_k[c], eta_k[c], sizeof(fftw_complex)*local_nx*ny);
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
            local_energy += 1;
        }
    }

    return energy;
}

void PhaseField::test() {
    take_fft(plan_forward);
    output_field(eta[0]);
    output_field(eta_k[0]);
    take_fft(plan_backward);
    normalize_field(eta);
    output_field(eta[0]);
}

