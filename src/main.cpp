

#include <iostream>

#include "pfc.h"


int main(int argc, char **argv) {
    
    PhaseField a(argc, argv);
    a.initialize_eta();

    a.test();

    a.calculate_energy();

}

