
# Phase-field crystal modelling with MPI

## Overview

The phase-field crystal (PFC) models liquid-solid phase transformations of crystals and can describe elasticity, plasticity and topological defects.

The program is a finite-difference based implementation of the PFC model. It also enables to perform elastic equilibration during the PFC time evolution. The L-BFGS algorithm is implemented to perform the elastic equilibration.

For more details, see the project report: http://kodu.ut.ee/~kreimre/phase_field_crystal_report.pdf

## Example results

The following videos show time evolution of the PFC model for two systems. First is a circular grain in a monocrystal. The second system consists of three different orientation seeds in a supercooled liquid and results in a crystal consisting of three grains.

![grain-contraction](https://github.com/eimrek/phase-field-crystal-mpi/blob/master/misc/img/grain_contraction.gif) ![seed-growth](https://github.com/eimrek/phase-field-crystal-mpi/blob/master/misc/img/seed_growth.gif)



