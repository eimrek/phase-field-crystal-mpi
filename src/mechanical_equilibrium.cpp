
#include <iostream>

#include "mechanical_equilibrium.h"

MechanicalEquilibrium::MechanicalEquilibrium(PhaseField* phase_field) {
    this->phase_field = phase_field;
}


void MechanicalEquilibrium::test() {
    phase_field->test();
}


void MechanicalEquilibrium::steepest_descent_fixed_dz() {
    double dz = 1.0;
    int max_iter = 10000;
    int check_freq = 100;
    double tolerance = 7.5e-9;

    for (int it = 0; it < max_iter; it++) {
        1;
    }

}

