import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print("Please supply a filename to convert.")
    exit

def print_comp(eta_):
    for i in range(nx):
        print("|", end="")
        for j in range(ny):
            print("%11.4e"%eta_[i, j], end="")
        print("|")

filepath = sys.argv[1]
imagename = filepath.replace("bin", "png")

print("Creating {0} from {1}".format(imagename, filepath))

nx = 512
ny = 512
nc = 3
flat = np.fromfile(filepath, dtype=np.complex128, count=-1, sep="")
eta = flat.reshape(nc, nx, ny)

val = np.abs(eta[0]) + np.abs(eta[1]) + np.abs(eta[2])
plt.pcolormesh(val)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xlim([0, nx])
plt.ylim([0, ny])
plt.savefig(imagename, dpi=400, bbox_inches="tight")
