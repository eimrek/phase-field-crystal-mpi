

#include <iostream>

#include "pfc.h"


int main(int argc, char **argv) {
    
    PhaseField a(argc, argv);
    a.initialize_eta();
    a.take_fft();
    a.output_field(a.get_eta(0));
    a.output_field(a.get_keta(0));
    a.take_ifft();
    a.output_field(a.get_eta(0));

    a.calculate_energy();

}

