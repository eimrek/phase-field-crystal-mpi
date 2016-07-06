
#include <fftw3-mpi.h>

#define sq3 1.7320508075688772935
#define PI 3.14159265358979323846
#define REAL 0
#define IMAG 1

class PhaseField {
    
    static const ptrdiff_t nx = 5, ny = 6;
    static const double dx = 2.0, dy = 2.0;

    static const double dt = 0.125;

    static const double q_vectors[][2];

    static const double bx = 1.0, bl = 0.95;
    static const double tt = 0.585, vv = 1.0;

    static const int nc = 3; //number of components
    
    fftw_complex **eta, **keta;
    fftw_plan *plan_forward, *plan_backward;

    fftw_complex **buffer, **buffer_k;
    fftw_plan *buffer_plan_f, *buffer_plan_b;



    ptrdiff_t local_nx, local_nx_start, alloc_local;

    int mpi_rank, mpi_size;

    double *k_x_values, *k_y_values;
    double **g_values;

    void calculate_k_values(double *k_values, int n, double d);
    void calculate_g_values(double **g_values);
    double calculate_aa();

public:
    void initialize_eta();
    void take_fft(fftw_plan *plan);
    void normalize_field(fftw_complex **field);

    fftw_complex* get_eta(int num);
    fftw_complex* get_keta(int num);

    void output_field(fftw_complex* field);

    double calculate_energy();

    PhaseField(int argc, char **argv);
    ~PhaseField();
    
    void test();
};
