
# Phase-field crystal modelling with MPI

## Overview

The phase-field crystal (PFC) models liquid-solid phase transformations of
crystals and can describe elasticity, plasticity and topological defects.

The program is a finite-difference based implementation of the PFC model. It
also enables to perform elastic equilibration during the PFC time evolution. The
L-BFGS algorithm is implemented to perform the elastic equilibration.

For more details, see the [project report](https://drive.google.com/file/d/1eDGmD9PXif09STVQw0Y3AcxmkMtJQfWx).

## Example results

The following videos show time evolution of the PFC model for two systems. First
is a circular grain in a monocrystal. The second system consists of three
different orientation seeds in a supercooled liquid and results in a crystal
consisting of three grains.

![grain-contraction][gc] ![seed-growth][sg]

## Usage

### Python

Python code is in the `python_code` directory. It requires a sub-directory named
"fig": create it before executing. Then, simply run

``` bash
$ python main.py
```

This will write an image, `fig/phi.png`, for your viewing pleasure.

### C++

You must compile the C++ code. To do so, run `make` from the top-level
directory. This will create the executable, `src/pfc`.

`pfc` requires a sub-directory named "output": create it before executing.
The default behavior is to write three files:

1. `initial_conf.bin`, representing the initial condition.
2. `eta50.bin`, representing the system after 400 timesteps.
3. `eta100.bin`, representing the system after 800 timesteps.

Each file contains the list of complex numbers, represented in 8-byte
double-precision, and written out in octal form. The fastest moving index is
*y*=[0,512], then *x*=[0,512], and finally *c*=[0,2]. To convert from octal to
ASCII, use the octal dump utility:

```bash
$ od -v -t fD file.bin
```

Perhaps a more useful method is to use the `misc/plot_binary_data.py` script.
Supply a `.bin` data file, and it will convert it to a `.png`:

``` bash
$ python misc/plot_binary_data.py output/initial_conf.bin
Creating output/initial_conf.png from output/initial_conf.bin
```

Note that if you change the grid size in the C++ source, you will have to change
the dimensions in `plot_binary_data.py` as well.

<!--References-->

[gc]: misc/img/grain_contraction.gif
[sg]: misc/img/seed_growth.gif
