#ifndef PFC_H
#define PFC_H

#include <complex>
#include <string>
#include <fftw3-mpi.h>


using namespace std;

#define sq3 1.7320508075688772935
#define PI 3.14159265358979323846

#define J complex<double>(0.0, 1.0)

#define REAL 0
#define IMAG 1

class PhaseField {
protected: 
    static const int nx, ny;
    static const double dx, dy;

    static const double dt;

    static const double q_vec[][2];

    static const double bx, bl;
    static const double tt, vv;

    static const int nc; //number of components
    

    ptrdiff_t alloc_local, local_nx, local_nx_start;

    int mpi_rank, mpi_size;

    double *k_x_values, *k_y_values;
    double **g_values;

    void calculate_k_values(double *k_values, int n, double d);
    void calculate_g_values(double **g_values);

    double dot_prod(const double* v1, const double* v2, int len);

    void memcopy_eta(complex<double> **eta_to, complex<double> **eta_from);
    
public:

    complex<double> **eta, **eta_k;
    fftw_plan *eta_plan_f, *eta_plan_b;

    complex<double> **eta_tmp, **eta_tmp_k;
    fftw_plan *eta_tmp_plan_f, *eta_tmp_plan_b;

    complex<double> **buffer, **buffer_k;
    fftw_plan *buffer_plan_f, *buffer_plan_b;

    double **grad_theta;


    void initialize_eta();
    void take_fft(fftw_plan *plan);
    void normalize_field(complex<double> **field);

    complex<double>* get_eta(int num);
    complex<double>* get_eta_k(int num);

    void output_field(complex<double>* field);

    double calculate_energy(complex<double> **eta_, complex<double> **eta_k_);
    void calculate_grad_theta(complex<double> **eta_, complex<double> **eta_k_);
    void calculate_nonlinear_part(int i, int j, complex<double> *compoenents,
            complex<double> **eta_);
    void overdamped_time_step();

    PhaseField(int mpi_rank, int mpi_size);
    ~PhaseField();
    
    void write_eta_to_file(string filename);
    void read_eta_from_file(string filename);

    void test();
};

#endif
