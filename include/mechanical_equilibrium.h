
#ifndef MECH_EQ_H
#define MECH_EQ_H

#include "pfc.h"

class MechanicalEquilibrium {
    PhaseField* phase_field;
    
public:
    MechanicalEquilibrium(PhaseField* phase_field);

    void test();

    void steepest_descent_fixed_dz();
};

#endif
