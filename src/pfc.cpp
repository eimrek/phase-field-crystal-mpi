
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
    
    k_x_values = (double*) malloc(sizeof(double)*nx);
    k_y_values = (double*) malloc(sizeof(double)*ny);
    calculate_k_values(k_x_values, nx, dx);
    calculate_k_values(k_y_values, ny, dy);

    g_values = (double**) malloc(sizeof(double*)*nc);
    for (int i = 0; i < nc; i++) {
        g_values[i] = (double*) malloc(sizeof(double)*nx*ny);
    }
    calculate_g_values(g_values);

    eta = (fftw_complex**) malloc(sizeof(fftw_complex*)*nc);
    keta = (fftw_complex**) malloc(sizeof(fftw_complex*)*nc);
    plan_forward = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);
    plan_backward = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);

    fftw_mpi_init(); 

    alloc_local = fftw_mpi_local_size_2d(nx, ny, MPI_COMM_WORLD,
            &local_nx, &local_nx_start);

    for (int i = 0; i < nc; i++) {
        eta[i] = fftw_alloc_complex(alloc_local);
        keta[i] = fftw_alloc_complex(alloc_local);
        plan_forward[i] = fftw_mpi_plan_dft_2d(nx, ny, eta[i], keta[i], MPI_COMM_WORLD,
            FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward[i] = fftw_mpi_plan_dft_2d(nx, ny, keta[i], eta[i], MPI_COMM_WORLD,
            FFTW_BACKWARD, FFTW_ESTIMATE);
    }

}

PhaseField::~PhaseField() {
    for (int i = 0; i < nc; i++) {
        fftw_free(eta[i]); fftw_free(keta[i]);
        fftw_destroy_plan(plan_forward[i]); 
        fftw_destroy_plan(plan_backward[i]); 
        free(g_values[i]);
    }
    free(eta); free(keta);
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

void PhaseField::take_fft() {
    for (int i = 0; i < 3; i++) {
        fftw_execute(plan_forward[i]);
    }
}

/*! Takes the inverse Fourier transform (with correct normalization)
 */
void PhaseField::take_ifft() {
    for (int i = 0; i < 3; i++) {
        fftw_execute(plan_backward[i]);
    }
    double scale = 1.0/(nx*ny);
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int n = 0; n < 3; n++) {
                eta[n][i*ny + j][REAL] *= scale;
                eta[n][i*ny + j][IMAG] *= scale;
            }
        }
    }
}

fftw_complex* PhaseField::get_eta(int num) {
    return eta[num];
}

fftw_complex* PhaseField::get_keta(int num) {
    return keta[num];
}

/*! Method that gathers all eta or keta data to root process and prints it out
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
        //std::cout << k_values[i] << " ";
    }
    //std::cout << std::endl;
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

double PhaseField::abs_squared(fftw_complex *field) {
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
                        
        }
    }
}


double PhaseField::calculate_energy() {
    // Buffer to hold (G_j eta_j)
    fftw_complex **g_eta_buffer = (fftw_complex**) malloc(sizeof(fftw_complex*)*nc);
    fftw_complex **g_eta_buffer_k = (fftw_complex**) malloc(sizeof(fftw_complex*)*nc);
    for (int c = 0; c < nc; c++) {
        g_eta_buffer[c] = fftw_malloc
    }

    free(g_eta_buffer);
}

