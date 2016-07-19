
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstring>
#include <array>


#include <mpi.h>
#include <fftw3-mpi.h>

#include "pfc.h"

// ---------------------------------------------------------------
// PARAMETERS
//
// initialization could also be done in header, if const->constexpr 

const int PhaseField::nx = 384;
const int PhaseField::ny = 384;

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

PhaseField::PhaseField(int mpi_rank, int mpi_size)
        : mpi_rank(mpi_rank), mpi_size(mpi_size), mech_eq(this) {

    fftw_mpi_init(); 
   
    // Allocate and calculate k values
    k_x_values = (double*) malloc(sizeof(double)*nx);
    k_y_values = (double*) malloc(sizeof(double)*ny);
    calculate_k_values(k_x_values, nx, dx);
    calculate_k_values(k_y_values, ny, dy);
    
    // Allocate etas and FFT plans and same for buffer values
    eta = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    eta_k = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    eta_plan_f = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);
    eta_plan_b = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);

    eta_tmp = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    eta_tmp_k = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    eta_tmp_plan_f = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);
    eta_tmp_plan_b = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);

    buffer = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    buffer_k = (complex<double>**) malloc(sizeof(complex<double>*)*nc);
    buffer_plan_f = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);
    buffer_plan_b = (fftw_plan*) malloc(sizeof(fftw_plan)*nc);



    alloc_local = fftw_mpi_local_size_2d(nx, ny, MPI_COMM_WORLD,
            &local_nx, &local_nx_start);

    // Allocate memory for G_j values and theta gradient
    g_values = (double**) malloc(sizeof(double*)*nc);
    grad_theta = (double**) malloc(sizeof(double*)*nc);
    for (int i = 0; i < nc; i++) {
        g_values[i] = (double*) malloc(sizeof(double)*local_nx*ny);
        grad_theta[i] = (double*) malloc(sizeof(double)*local_nx*ny);
    }
    calculate_g_values(g_values);


    for (int i = 0; i < nc; i++) {
        eta[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
        eta_k[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
        eta_plan_f[i] = fftw_mpi_plan_dft_2d(nx, ny,
                reinterpret_cast<fftw_complex*>(eta[i]),
                reinterpret_cast<fftw_complex*>(eta_k[i]),
                MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
        eta_plan_b[i] = fftw_mpi_plan_dft_2d(nx, ny,
                reinterpret_cast<fftw_complex*>(eta_k[i]),
                reinterpret_cast<fftw_complex*>(eta[i]),
                MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);

        eta_tmp[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
        eta_tmp_k[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
        eta_tmp_plan_f[i] = fftw_mpi_plan_dft_2d(nx, ny,
                reinterpret_cast<fftw_complex*>(eta_tmp[i]),
                reinterpret_cast<fftw_complex*>(eta_tmp_k[i]),
                MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
        eta_tmp_plan_b[i] = fftw_mpi_plan_dft_2d(nx, ny,
                reinterpret_cast<fftw_complex*>(eta_tmp_k[i]),
                reinterpret_cast<fftw_complex*>(eta_tmp[i]),
                MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);

        buffer[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
        buffer_k[i] = reinterpret_cast<complex<double>*>(fftw_alloc_complex(alloc_local));
        buffer_plan_f[i] = fftw_mpi_plan_dft_2d(nx, ny,
                reinterpret_cast<fftw_complex*>(buffer[i]),
                reinterpret_cast<fftw_complex*>(buffer_k[i]),
                MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
        buffer_plan_b[i] = fftw_mpi_plan_dft_2d(nx, ny,
                reinterpret_cast<fftw_complex*>(buffer_k[i]),
                reinterpret_cast<fftw_complex*>(buffer[i]),
                MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
    }

}

PhaseField::~PhaseField() {
    for (int i = 0; i < nc; i++) {
        fftw_free(eta[i]); fftw_free(eta_k[i]);
        fftw_destroy_plan(eta_plan_f[i]); fftw_destroy_plan(eta_plan_b[i]); 

        fftw_free(buffer[i]); fftw_free(buffer_k[i]);
        fftw_destroy_plan(buffer_plan_f[i]); fftw_destroy_plan(buffer_plan_b[i]); 

        free(g_values[i]);
    }
    free(eta); free(eta_k);
    free(eta_plan_f); free(eta_plan_b);

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

    complex<double> *field_total = (complex<double>*)
            fftw_malloc(sizeof(complex<double>)*nx*ny);

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
    
        std::cout << std::endl;
    }
    fftw_free(field_total);
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


void PhaseField::memcopy_eta(complex<double> **eta_to, complex<double> **eta_from) {
    for (int c = 0; c < nc; c++) {
        std::memcpy(eta_to[c], eta_from[c], sizeof(complex<double>)*local_nx*ny);
    }
}


/*! Method to calculate energy.
 *
 *  NB: This method assumes that eta_k is set beforehand.
 *  Takes 1 fft
 */
double PhaseField::calculate_energy(complex<double> **eta_, complex<double> **eta_k_) {
    // will use the member variable buffer_k to hold (G_j eta_j)_k
    memcopy_eta(buffer_k, eta_k_);

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
            complex<double> eta0=eta_[0][i*ny+j]; complex<double> buf0=buffer[0][i*ny+j];
            complex<double> eta1=eta_[1][i*ny+j]; complex<double> buf1=buffer[1][i*ny+j];
            complex<double> eta2=eta_[2][i*ny+j]; complex<double> buf2=buffer[2][i*ny+j];
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
    /*
    if (mpi_rank == 0) {
        printf("Energy: %.16e\n", energy);
    }
    */
    return energy;
}

/*! Method, that calculates the three components of the nonlinear part
 *
 *  The components will be saved to components (memory must be allocated before)
 *  and they correspond to real space indices (coordinates) of i, j
 */
void PhaseField::calculate_nonlinear_part(int i, int j, complex<double> *components,
        complex<double> **eta_) {
    complex<double> eta0 = eta_[0][i*ny+j];
    complex<double> eta1 = eta_[1][i*ny+j];
    complex<double> eta2 = eta_[2][i*ny+j];
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
    memcopy_eta(buffer, eta);

    // allocate memory for the nonlinear part
    complex<double> *nonlinear_part = (complex<double>*) malloc(sizeof(complex<double>)*nc);

    // buffer contains eta, let's subtract dt*(nonlinear part)
    // to get the numerator in the OD time stepping scheme (in real space)
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            calculate_nonlinear_part(i, j, nonlinear_part, eta);
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
    take_fft(eta_plan_b);
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
 *  The resulting gradient will be stored in "grad_theta"
 *  NB: required eta_k to be set
 *  Takes 1 fft
 */
void PhaseField::calculate_grad_theta(complex<double> **eta_, complex<double> **eta_k_) {
    // will use the member variable buffer_k to hold (G_j^2 eta_j)_k
    memcopy_eta(buffer_k, eta_k_);

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
           calculate_nonlinear_part(i, j, nonlinear_part, eta_);
           for (int c = 0; c < nc; c++) {
                complex<double> var_f_eta = (bl-bx)*eta_[c][i*ny+j] + bx*buffer[c][i*ny+j]
                                            + nonlinear_part[c];
                im[c] = imag(conj(eta_[c][i*ny+j])*var_f_eta);
           }
           for (int c = 0; c < nc; c++) {
                double dtheta = dot_prod(q_vec[c], q_vec[0], 2)*im[0]
                               + dot_prod(q_vec[c], q_vec[1], 2)*im[1]
                               + dot_prod(q_vec[c], q_vec[2], 2)*im[2];
                grad_theta[c][i*ny+j] = dtheta;
           }
        }
    }

    free(nonlinear_part);
    free(im);
}

/*! Method, that writes current eta to a binary file
 *
 *  The data is structured as follows:
 *  8 byte doubles, alternating real and imaginary parts,
 *  if eta[c][i*ny+j], then fastest moving index is j, then i and finally c
 */
void PhaseField::write_eta_to_file(string filepath) {


    // delete old file
    if (mpi_rank == 0) MPI_File_delete(filepath.c_str(), MPI_INFO_NULL);

    MPI_File mpi_file;
    int rcode = MPI_File_open(MPI_COMM_WORLD, filepath.c_str(),
            MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_file);

    if (rcode != MPI_SUCCESS)
        cerr << "Error: couldn't open file" << endl;
    for (int c = 0; c < nc; c++) {
        MPI_Offset offset = mpi_rank*local_nx*ny*sizeof(double)*2 + c*nx*ny*sizeof(double)*2;
        rcode = MPI_File_set_view(mpi_file, offset,
                                    MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    
        if(rcode != MPI_SUCCESS)
            cerr << "Error: couldn't set file process view" << endl;
    
        rcode = MPI_File_write(mpi_file, eta[c], local_nx*ny*2,
                    MPI_DOUBLE, MPI_STATUS_IGNORE);

        if(rcode != MPI_SUCCESS)
            cerr << "Error: couldn't write file" << endl;
    }

    MPI_File_close(&mpi_file);
}

/*! Reads eta from a binary file written by write_eta_to_file
 *
 */
void PhaseField::read_eta_from_file(string filepath) {

    MPI_File mpi_file;
    int rcode = MPI_File_open(MPI_COMM_WORLD, filepath.c_str(), MPI_MODE_RDWR,
            MPI_INFO_NULL, &mpi_file);

    if (rcode != MPI_SUCCESS)
        cerr << "Error: couldn't open file" << endl;

    for (int c = 0; c < nc; c++) {
        MPI_Offset offset = mpi_rank*local_nx*ny*sizeof(double)*2 + c*nx*ny*sizeof(double)*2;
        rcode = MPI_File_set_view(mpi_file, offset,
                                    MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    
        if(rcode != MPI_SUCCESS)
            cerr << "Error: couldn't set file process view" << endl;
    
        rcode = MPI_File_read(mpi_file, eta[c], local_nx*ny*2,
                    MPI_DOUBLE, MPI_STATUS_IGNORE);

        if(rcode != MPI_SUCCESS)
            cerr << "Error: couldn't read file" << endl;
    }
    MPI_File_close(&mpi_file);
}


double PhaseField::calculate_radius() {
    int line_x = nx/2 - local_nx_start;

    int argmin1 = 0, argmin2 = 0;
    double phi_min = 0.0;

    if (line_x >= 0 && line_x < local_nx) {
        for (int j = 0; j < ny; j++) {
            double phi = 0.0;
            for (int c = 0; c < nc; c++)
                phi += abs(eta[c][line_x*ny + j]);
            if (j == 0 || j == ny/2)
                phi_min = phi;
            // if in first half
            if (j < ny/2) {
                if (phi <= phi_min) {
                    phi_min = phi;
                    argmin1 = j;
                }           
            } else {
                if (phi <= phi_min) {
                    phi_min = phi;
                    argmin2 = j;
                }           
            }
        }
    }
    double radius = abs(argmin2 - argmin1)*dy; 
    MPI_Allreduce(&radius, &radius, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return radius;
}

/*! Method, does the initial setup to run the circle calculations from scratch
 *
 */
void PhaseField::start_calculations() {

    string path = "./output/testrun/";
    string run_info_filename = "run_info.txt";
    
    Time::time_point time_start = Time::now();

    // initialize eta and eta_k
    initialize_eta();
    take_fft(eta_plan_f);

    double energy = calculate_energy(eta, eta_k);
    double radius = calculate_radius();
    if (mpi_rank == 0)
        printf("Initial state - energy: %.16e; radius: %5.1f\n", energy, radius);

    int initial_od_steps = 10000;

    for (int it = 0; it < initial_od_steps; it++) {
        overdamped_time_step();
    }

    energy = calculate_energy(eta, eta_k);
    radius = calculate_radius();

    if (mpi_rank == 0) {
        double total_dur = std::chrono::duration<double>(Time::now()-time_start).count();
        printf("%d OD steps: energy %.16e; radius: %5.1f; total_time: %7.1f\n",
                initial_od_steps, energy, radius, total_dur);
    }

    int num_fft = 0;
    int meq_iter = mech_eq.accelerated_steepest_descent_adaptive_dz(&num_fft);
    energy = calculate_energy(eta, eta_k);
    double total_dur = std::chrono::duration<double>(Time::now()-time_start).count();
    if (mpi_rank == 0) {
        printf("Initial mech. eq.: energy %.16e; iter: %5d; total_time: %7.1f\n",
                energy, meq_iter, total_dur);
    }


    run_calculations(initial_od_steps, total_dur, num_fft, path, run_info_filename);
}


void PhaseField::run_calculations(int init_it, double time_so_far, int total_ffts,
        string path, string run_info_filename) {

    Time::time_point time_start = Time::now();
    Time::time_point time_var = Time::now();

    // Start repetitions of overdamped steps and mechanical equilibrium
    int repetitions = 10000;
    int od_steps = 80;

    int save_freq = 100;
    FILE * run_info_file;

    int it = init_it; // total over-damped iteration counter
    double last_radius = -1.0;

    for (int rep = 0; rep < repetitions; rep++) {
        time_var = Time::now();
        // Over-damped timesteps
        for (int ts = 0; ts < od_steps; ts++) {
            overdamped_time_step();
        }
        it += od_steps;
        double od_dur = std::chrono::duration<double>(Time::now()-time_var).count();
        // Mechanical equilibration
        int num_fft = 0;
        int meq_iter = mech_eq.accelerated_steepest_descent_adaptive_dz(&num_fft);
        total_ffts += num_fft;
        double meq_dur = std::chrono::duration<double>(Time::now()-time_var).count()
                           - od_dur;
        double energy = calculate_energy(eta, eta_k);
        double radius = calculate_radius();
        double total_dur = std::chrono::duration<double>(Time::now()-time_start).count()
                            + time_so_far;
        if (mpi_rank == 0) {
            // Print run information
            printf("it: %5d; stime: %7.1f; energy: %.16e; radius: %5.1f; od_time: %4.1f; "
                   "meq_iter: %4d; meq_time: %5.1f; total_ffts: %d, total_time: %7.1f\n",
                   it, it*dt, energy, radius, od_dur,
                   meq_iter, meq_dur, total_ffts, total_dur);
            // Save run information also to a file
            run_info_file = fopen((path+run_info_filename).c_str(), "a");
            fprintf(run_info_file, "%d %.1f %.16e %.1f %.1f %d %.1f %d %.1f\n",
                    it, it*dt, energy, radius, od_dur,
                    meq_iter, meq_dur, total_ffts, total_dur);
            fclose(run_info_file);
        }
        if (last_radius > 0 && radius > last_radius) {
            std::stringstream sstream;
            sstream << std::fixed << std::setprecision(0) << it*dt;
            write_eta_to_file(path+"eta_"+sstream.str()+".bin");
            cout << "Radius increased. Probably reached the end of calculations. Ending."
                << endl;
            break;
        }
        if (rep % save_freq == 0) {
            std::stringstream sstream;
            sstream << std::fixed << std::setprecision(0) << it*dt;
            write_eta_to_file(path+"eta_"+sstream.str()+".bin");
        }
    }
}


void PhaseField::continue_calculations() {

    string path = "./output/testrun/";
    string run_info_filename = "run_info.txt";

    string continue_from_file = "eta_10.bin";
    double continue_stime = 10.0;
    
    // Load eta
    read_eta_from_file(path+continue_from_file);
    if (mpi_rank == 0)
        cout << "Loaded eta from file: " << path+continue_from_file << endl;

    FILE * run_info_file = fopen((path+run_info_filename).c_str(), "r");

    std::vector< std::array<double, 9> > data;

    int init_it, meq_iter, total_ffts;
    double stime, en, rad, odd, meqd, total_dur;
    // Read the data until the starting point from run_info_file
    while (fscanf(run_info_file, "%d %lf %lf %lf %lf %d %lf %d %lf\n",
                    &init_it, &stime, &en, &rad, &odd, &meq_iter,
                    &meqd, &total_ffts, &total_dur) != EOF) {
        std::array<double, 9> line{
            {double(init_it), stime, en, rad, odd, double(meq_iter), meqd,
                double(total_ffts), total_dur}
        };
        data.push_back(line);
        if (abs(stime-continue_stime) < 0.1) break;
    }
    
    fclose(run_info_file);
    run_info_file = fopen((path+run_info_filename).c_str(), "w");
    // Overwrite the data with only the relevant part
    for (unsigned int ln = 0; ln < data.size(); ln++) {
        fprintf(run_info_file, "%d %.1f %.16e %.1f %.1f %d %.1f %d %.1f\n",
                int(data[ln][0]), data[ln][1], data[ln][2],
                data[ln][3], data[ln][4], int(data[ln][5]),
                data[ln][6], int(data[ln][7]), data[ln][8]);
    }
    fclose(run_info_file);

    if (mpi_rank == 0)
        cout << "Read info from " << run_info_filename
             << " and cleaned redundant entries." << endl; 

    run_calculations(init_it, total_dur, total_ffts, path, run_info_filename);
}


void PhaseField::test() {

    read_eta_from_file("./output/testrun/eta_20.bin");

    for (int it = 0; it < 80; it++) {
        overdamped_time_step();
    }

    int num_fft = 0;
    int iter = mech_eq.accelerated_steepest_descent_adaptive_dz(&num_fft);

    cout << num_fft << endl;
    
}


