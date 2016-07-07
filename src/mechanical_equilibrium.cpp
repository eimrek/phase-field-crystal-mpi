
#include <iostream>

#include "mechanical_equilibrium.h"

MechanicalEquilibrium::MechanicalEquilibrium(PhaseField* phase_field) {
    this->phase_field = phase_field;
}


void MechanicalEquilibrium::test() {
    phase_field->test();
}


