"""
PFC code in python

- Kristjan Eimre
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal
import scipy.ndimage
import equilibrium_algorithms
import equilibrium_algorithms_2


# number of cells in the grid and cell dimensions
# 384
# 1536
nx = 384
ny = 384
dx = 2.0
dy = 2.0

dt = 0.0125

# parameters ---------------------------------------------------
# one mode approximation lowest order reciprocal lattice vectors
q_vectors = np.array([
    [-np.sqrt(3.0)/2, -1/2],
    [0, 1],
    [np.sqrt(3.0)/2, -1/2]
])

bx = 1.0
bl = bx-0.05
tt = 0.585
vv = 1.0

# -------------------------------------------------------------
# calculate the k values corresponding to bins in k space
k_x_values = 2*np.pi*np.fft.fftfreq(nx, dx)
k_y_values = 2*np.pi*np.fft.fftfreq(ny, dy)

# -------------------------------------------------------------
# G_j coefficients and the denominator of the (over-damped) numerical scheme in k space
g_values = np.zeros((3, nx, ny))
g_values_fd = np.zeros((3, nx, ny))
denominator_k = np.zeros((3, nx, ny))
# -------------------------------------------------------------
# finite difference kernels for finding theta gradients
fd_kernels = []
# -------------------------------------------------------------


def init_state_circle(eta):
    """
    Initialize the state of the phase field
    """

    # Keep amplitude constant, but rotate the phase by some angle for a circle in the middle
    angle = 0.0872665
    amplitude = 0.10867304595992146

    theta = np.zeros((3, nx, ny))

    for i in range(nx):
        for j in range(ny):
            # If inside the circle with a radius of 0.25*nx*dx
            if (i+1-nx/2)**2*dx**2 + (j+1-ny/2)**2*dy**2 <= (0.25*nx*dx)**2:
                for n in range(3):
                    # apply a rotation
                    theta = ((q_vectors[n, 0]*np.cos(angle) + q_vectors[n, 1]*np.sin(angle) - q_vectors[n, 0])
                                * (i+1 - nx/2) * dx
                            + (-q_vectors[n, 0]*np.sin(angle) + q_vectors[n, 1]*np.cos(angle) - q_vectors[n, 1])
                                * (j+1 - ny/2) * dy
                            )
                    eta[n, i, j] = amplitude * np.exp(1j*theta)
            # if outside the circle, no rotation
            else:
                eta[:, i, j] = amplitude


def init_state_seed(eta):

    amplitude = 0.10867304595992146
    seed_radius = 0.15*nx*dx

    seed_pos = [(0.3, 0.5), (0.7, 0.5)]
    seed_angles = [0.0, 0.2]

    for i in range(nx):
        for j in range(ny):
            for c in range(3):
                eta[c, i, j] = 0.0
                for sd, ang in zip(seed_pos, seed_angles):
                    # current coordinates
                    x = i*dx
                    y = j*dy

                    # seed center coordinates
                    x_c = (nx-1)*sd[0]*dx
                    y_c = (ny-1)*sd[1]*dy

                    # distance from center
                    center_dist = np.sqrt((x-x_c)**2 + (y-y_c)**2)
                    rd = center_dist/seed_radius

                    # apply a rotation around point (x_c, y_c) by angle "ang"
                    rot_mat = np.array([[np.cos(ang)-1, -np.sin(ang)],
                                        [np.sin(ang), np.cos(ang)-1]])
                    row_vec = np.array([x-x_c, y-y_c])
                    theta = np.dot(q_vectors[c], np.dot(rot_mat, row_vec))

                    eta[c, i, j] += amplitude * np.exp(1j*theta) / (rd**12+1)



def init_state_seed_test(eta):
    amplitude = 0.10867304595992146
    seed_radius = 0.02*nx*dx
    for i in range(nx):
        for j in range(ny):
            for c in range(3):
                eta[c, i, j] = 0.0
                # distance from center
                center_dist = np.sqrt((i-(nx-1)/2)**2*dx**2 + (j-(ny-1)/2)**2*dy**2)
                rd = center_dist/seed_radius
                if rd < 3.0:  # <- !!! Bug doesn't occur without this check !!!
                    eta[c, i, j] += amplitude / (rd**4+1)


def calculate_coefficients():
    """
    Calculates the G_j expression and the denominator of the numerical scheme in k space
    """
    for i in range(nx):
        for j in range(ny):
            k_square = k_x_values[i]**2+k_y_values[j]**2
            cct = -k_square - 2*(q_vectors[:,0]*k_x_values[i] + q_vectors[:,1]*k_y_values[j])
            g_values[:, i, j] = cct
            denominator_k[:, i, j] = 1 + dt*((bl-bx)+bx*cct**2)


def calculate_coefficients_mat():
    """
    Calculates the G_j expression and the denominator of the numerical scheme in k space
    """
    k_x_matrix, k_y_matrix = np.meshgrid(k_x_values,k_y_values)
    k_y_matrix = np.transpose(k_y_matrix)
    k_x_matrix = np.transpose(k_x_matrix)
    for i in range(3):
        g_values[i] = -(k_x_matrix**2+k_y_matrix**2) - 2*(q_vectors[i, 0]*k_x_matrix + q_vectors[i, 1]*k_y_matrix)
    denominator_k[:] = 1 + dt*((bl-bx)+bx*g_values**2)


def calculate_coefficients_tile():
    """
    Calculates the G_j expression and the denominator of the numerical scheme in k space
    """
    k_x_matrix=np.transpose(np.tile(k_x_values,(ny,1)))
    k_y_matrix=np.tile(k_y_values,(nx,1))
    for i in range(3):
        g_values[i] = -(k_x_matrix**2+k_y_matrix**2) - 2*(q_vectors[i, 0]*k_x_matrix + q_vectors[i, 1]*k_y_matrix)
        #g_values_fd[i] = 1.0/(6*dx**2)*(16*np.cos(k_x_matrix*dx)-np.cos(2*k_x_matrix*dx)-15)
        #g_values_fd[i] += 1.0/(6*dy**2)*(16*np.cos(k_y_matrix*dy)-np.cos(2*k_y_matrix*dy)-15)
        #g_values_fd[i] += -(q_vectors[i, 0]/(3*dx)*(8*np.sin(k_x_matrix*dx)-np.sin(2*k_x_matrix*dx)))
        #g_values_fd[i] += -(q_vectors[i, 1]/(3*dy)*(8*np.sin(k_y_matrix*dy)-np.sin(2*k_y_matrix*dy)))
    denominator_k[:] = 1 + dt*((bl-bx)+bx*g_values**2)


def calculate_energy(eta, eta_k):
    """
    Calculates the energy of the phase field (eq. (2.89) from Vili's thesis)
    :return: energy
    """

    # calculate the A^2
    aa = 2*(np.abs(eta[0])**2 + np.abs(eta[1])**2 + np.abs(eta[2])**2)

    # buffer to hold the result of (G_j eta_j)
    # (G_j eta_j) will be evaluated by going to k space
    g_eta_buffer_k = np.copy(eta_k)

    # Add the G_j expression in k space to the buffer
    g_eta_buffer_k = g_eta_buffer_k*g_values

    # go back to real space for (G_j eta_j)
    g_eta_buffer = np.zeros((3, nx, ny), dtype=np.complex128)
    for j in range(3):
        g_eta_buffer[j] = np.fft.ifft2(g_eta_buffer_k[j])

    num_cells = nx*ny  # divide the total energy by this to get the energy density

    # Integrate over space to get the energy and divide by num cells to get density
    energy = np.sum(
                aa*(bl-bx)/2 + (3/4)*vv*aa**2
                - 4*tt*np.real(eta[0]*eta[1]*eta[2])
                + bx*(np.abs(g_eta_buffer[0])**2 + np.abs(g_eta_buffer[1])**2 + np.abs(g_eta_buffer[2])**2)
                - 3/2*vv*((np.abs(eta[0]))**4 + (np.abs(eta[1]))**4 + (np.abs(eta[2]))**4)
                )/num_cells

    return energy

def calculate_energy_fd(eta, eta_k):
    """
    Calculates the energy of the phase field
    :return: energy
    """

    # calculate the A^2
    aa = 2*(np.abs(eta[0])**2 + np.abs(eta[1])**2 + np.abs(eta[2])**2)

    # buffer to hold the result of (G_j eta_j)
    # (G_j eta_j) will be evaluated by going to k space
    g_eta_buffer_k = np.copy(eta_k)

    # Add the G_j expression in k space to the buffer
    g_eta_buffer_k = g_eta_buffer_k*g_values_fd

    # go back to real space for (G_j eta_j)
    g_eta_buffer = np.zeros((3, nx, ny), dtype=np.complex128)
    for j in range(3):
        g_eta_buffer[j] = np.fft.ifft2(g_eta_buffer_k[j])

    num_cells = nx*ny  # divide the total energy by this to get the energy density

    # Integrate over space to get the energy and divide by num cells to get density
    energy = np.sum(
                aa*(bl-bx)/2 + (3/4)*vv*aa**2
                - 4*tt*np.real(eta[0]*eta[1]*eta[2])
                + bx*(np.abs(g_eta_buffer[0])**2 + np.abs(g_eta_buffer[1])**2 + np.abs(g_eta_buffer[2])**2)
                - 3/2*vv*((np.abs(eta[0]))**4 + (np.abs(eta[1]))**4 + (np.abs(eta[2]))**4)
                )/num_cells

    return energy


def calc_nonlinear_part(eta):
    """
    Calculates the part of (dF/deta*_j) which doesn't contain the derivatives (gradients/laplacians)
    (Note: "\Delta B \eta" is not calculated here)
    """
    # calculate the A^2
    aa = 2*(np.abs(eta[0])**2 + np.abs(eta[1])**2 + np.abs(eta[2])**2)

    var_f_eta_noderiv = np.zeros((3, nx, ny), dtype=np.complex128)

    var_f_eta_noderiv[0] = (3*vv*(aa-np.abs(eta[0])**2)*eta[0] - 2*tt*np.conj(eta[1])*np.conj(eta[2]))
    var_f_eta_noderiv[1] = (3*vv*(aa-np.abs(eta[1])**2)*eta[1] - 2*tt*np.conj(eta[0])*np.conj(eta[2]))
    var_f_eta_noderiv[2] = (3*vv*(aa-np.abs(eta[2])**2)*eta[2] - 2*tt*np.conj(eta[1])*np.conj(eta[0]))

    return var_f_eta_noderiv


def time_step(eta, eta_k):
    """
    Makes a time step for etas
    """
    # Calculate the part of (dF/deta*_j) which doesn't contain the derivatives
    nonlinear_part = calc_nonlinear_part(eta)

    # numerator in the time stepping scheme for three etas (in real and k space)
    numerator = eta - dt*nonlinear_part
    numerator_k = np.zeros((3, nx, ny), dtype=np.complex128)

    for i in range(3):
        numerator_k[i] = np.fft.fft2(numerator[i])

    # the denominator is the part with derivatives, can be expressed easily in k space
    # the resulting value is the next eta of the numerical scheme in k space
    eta_k[:] = numerator_k/denominator_k

    # and the final eta
    for i in range(3):
        eta[i] = np.fft.ifft2(eta_k[i])


def calc_grad_theta(eta, eta_k):
    """
    Calculates the gradient of the function that needs to be minimized for theta dynamics for elastic equilibrium
    Based on Eq. (3.38) and (3.44) of Vili's thesis
    """

    # First, evaluate (G_j^2 eta_j) in k space
    # and return to real space
    g2_eta_buffer_k = g_values**2*eta_k
    g2_eta_buffer = np.fft.ifft2(g2_eta_buffer_k)

    # Calculate the part of (dF/deta*_j) which doesn't contain the derivatives
    nonlinear_part = calc_nonlinear_part(eta)

    # variation of f wrt eta (Eq. (3.38) in Vili's dissertation)
    var_f_eta = (bl-bx)*eta + bx*g2_eta_buffer + nonlinear_part

    # the fake time derivative for thetas
    im = np.imag(np.conj(eta)*var_f_eta)

    dtheta = np.zeros((3, nx, ny))
    for i in range(3):
        dtheta[i] = (np.dot(q_vectors[i], q_vectors[0])*im[0]
                      + np.dot(q_vectors[i], q_vectors[1])*im[1]
                      + np.dot(q_vectors[i], q_vectors[2])*im[2])
    return dtheta


def calculate_fd_kernels():
    # Find the kernel for each of the eta components
    laplacian_kernel = np.zeros((5, 5), dtype=np.complex128)
    # Order h^4 accuracy
    laplacian_kernel[:, 2] += 1.0/(12*dx**2)*np.array([-1.0, 16.0, -30.0, 16.0, -1.0])
    laplacian_kernel[2, :] += 1.0/(12*dy**2)*np.array([-1.0, 16.0, -30.0, 16.0, -1.0])

    # Order h^2 (?) accuracy
    #laplacian_kernel[:, 2] += 1.0/(dx**2)*np.array([0.0, 1.0, -2.0, 1.0, 0.0])
    #laplacian_kernel[2, :] += 1.0/(dy**2)*np.array([0.0, 1.0, -2.0, 1.0, 0.0])

    for j in range(3):
        qj_kernel = np.zeros((5,5))
        qj_kernel[:, 2] += q_vectors[j, 0]/(12*dx)*np.array([-1.0, 8.0, 0.0, -8.0, 1.0])
        qj_kernel[2, :] += q_vectors[j, 1]/(12*dy)*np.array([-1.0, 8.0, 0.0, -8.0, 1.0])
        #qj_kernel = np.array([[0.0, -1.0*q_vectors[j, 1]/dy, 0.0],
        #                   [-1.0*q_vectors[j, 0]/dx, 0.0, 1.0*q_vectors[j, 0]/dx],
        #                   [0.0, 1.0*q_vectors[j, 1]/dy, 0.0]])

        # First initialize as the biharmonic operator's fd kernel
        kernel = scipy.signal.convolve2d(laplacian_kernel, laplacian_kernel)
        # add the middle term of G_j
        kernel += 4.0j*scipy.signal.convolve2d(laplacian_kernel, qj_kernel)
        # And the last one
        kernel += -4.0*scipy.signal.convolve2d(qj_kernel, qj_kernel)
        fd_kernels.append(kernel)


def calc_grad_theta_fd(eta, eta_k):
    """
    Calculates the gradient of the thetas using finite differences
    """

    # Acutally, its slow with kernel, use k-space formulas...
    # evaluate (G_j^2 eta_j)
    #g2_eta_buffer = np.zeros((3, nx, ny), dtype=np.complex128)
    #for j in range(3):
    #    g2_eta_buffer[j, :] = scipy.signal.convolve2d(eta[j], fd_kernels[j], mode='same', boundary='wrap')

    # First, evaluate (G_j^2 eta_j) in k space
    # and return to real space
    g2_eta_buffer_k = g_values_fd**2*eta_k
    g2_eta_buffer = np.fft.ifft2(g2_eta_buffer_k)

    # Calculate the part of (dF/deta*_j) which doesn't contain the derivatives
    nonlinear_part = calc_nonlinear_part(eta)

    # variation of f wrt eta (Eq. (3.38) in Vili's dissertation)
    var_f_eta = (bl-bx)*eta + bx*g2_eta_buffer + nonlinear_part

    # the fake time derivative for thetas
    im = np.imag(np.conj(eta)*var_f_eta)

    dtheta = np.zeros((3, nx, ny))
    for i in range(3):
        dtheta[i] = (np.dot(q_vectors[i], q_vectors[0])*im[0]
                      + np.dot(q_vectors[i], q_vectors[1])*im[1]
                      + np.dot(q_vectors[i], q_vectors[2])*im[2])
    return dtheta


def mechanical_equilibrium(eta, eta_k):

    #equilibrium_algorithms.steepest_descent_fixed_dz(eta, calculate_energy, calc_grad_theta, nx, ny)
    #equilibrium_algorithms.steepest_descent_fixed_dz_finite_diff(eta, calculate_energy, calc_grad_theta, calculate_energy_fd, calc_grad_theta_fd, nx, ny)
    #equilibrium_algorithms.steepest_descent_exp_search(eta, calculate_energy, calc_grad_theta, nx*ny)

    #equilibrium_algorithms.steepest_descent_bin_search(eta, calculate_energy, calc_grad_theta, nx*ny)
    #equilibrium_algorithms.conjugate_gradient(eta, calculate_energy, calc_grad_theta, nx, ny)

    #equilibrium_algorithms.accelerated_steepest_descent(eta, calculate_energy, calc_grad_theta, nx*ny)
    #equilibrium_algorithms_2.conjugate_gradient_fixed_dz(eta, calculate_energy, calc_grad_theta, nx, ny)

    #iter = equilibrium_algorithms.accelerated_steepest_descent_adaptive_dz(eta, calculate_energy, calc_grad_theta, nx*ny)
    #equilibrium_algorithms.accelerated_steepest_descent_adaptive_dz_finite_diff(eta, calculate_energy, calc_grad_theta_fd, nx*ny)

    #equilibrium_algorithms_2.lbfgs(eta, calculate_energy, calc_grad_theta, nx, ny)

    iter = equilibrium_algorithms_2.lbfgs_enh(eta, calculate_energy, calc_grad_theta, nx, ny)

    #equilibrium_algorithms_2.adadelta(eta, calculate_energy, calc_grad_theta, nx, ny)

    eta_k[:] = np.fft.fft2(eta)
    return iter


def init_plot_3(eta):
    fig = plt.figure(figsize=(20,4))
    ax = []
    quad = []
    for i in range(3):
        ax.append(plt.subplot(1, 3, i+1))
        quad.append(plt.pcolormesh(np.abs(eta[i])))
        plt.colorbar(quad[i], ax=ax[i])
        ax[i].set_xlim([0, nx])
        ax[i].set_ylim([0, ny])
    plt.ion()
    plt.show()
    return ax, quad


def init_plot(eta):
    plt.subplot(111)
    quad = plt.pcolormesh(1/3*(abs(eta[0])+abs(eta[1])+abs(eta[2])))
    plt.colorbar()
    plt.xlim([0, nx])
    plt.ylim([0, ny])
    plt.ion()
    plt.show()
    return quad


def update_plot(quad, stime, eta, path, radius=-1.0):
    a = (1/3*(np.abs(eta[0])+np.abs(eta[1])+np.abs(eta[2]))).ravel()
    quad.set_array(a)
    if radius > 0.0:
        plt.title("ts %.1f, radius %.0f" % (stime, radius))
    else:
        plt.title("ts %.1f" % stime)
    plt.draw()
    plt.savefig(path+"time%.0f.png" % stime, dpi=300)


def defect_radius(eta):
    """
    Calculates the defect radius along centered vertical line
    """
    phi_line = (np.abs(eta[0])+np.abs(eta[1])+np.abs(eta[2]))[nx//2]
    side1 = np.argmin(phi_line[0:ny//2])
    side2 = np.argmin(phi_line[ny//2:])+ny//2
    return (side2-side1)*dy


def start_calculation():

    path = "./data/main_run/no_mech_eq/"

    # eta holds the three complex amplitudes (eta_k = eta in k space)
    eta = np.zeros((3, nx, ny), dtype=np.complex128)

    # Calculate some values that are the same for each time step
    # and should be calculated only once
    time_start = time.time()
    calculate_coefficients_tile()
    print("Coefficient calculation: %.3f s" % (time.time()-time_start))

    start_iterations = 10000

    # Load the starting state with 10000 O-D iterations done
    eta = np.load("./data/over_damped_%d.npy" % start_iterations)
    eta_k = np.fft.fft2(eta)
    print("Initial energy: %.16e" % calculate_energy(eta, eta_k))

    # start the global clock
    time_start = time.time()

    # Do the initial mechanical equilibrium
    meq_iter = mechanical_equilibrium(eta, eta_k)
    energy = calculate_energy(eta, eta_k)
    total_time = time.time()-time_start
    radius = defect_radius(eta)
    print("Initial mechanical eq. done. Energy: %.16e; iterations: %d; radius: %.1f; time taken: %.1f s" %
          (energy, meq_iter, radius, total_time))
    simulation_data = [[start_iterations, start_iterations*dt, energy, radius, meq_iter, total_time]]

    #np.savez("./data/mech_run/init_od_mech_eq", np.array(simulation_data), eta)
    run_calculation(simulation_data, eta, eta_k, path)


def run_calculation(simulation_data, eta, eta_k, datapath):
    """
    Function, that will resume the simulation based on the current state (last row of simulation_data)
    """
    iterations, stime, energy, radius, meq_iter, total_time = simulation_data[-1]

    time_start = time.time() - total_time

    # Initialize the plot for real time plotting
    quad = init_plot(eta)

    # plot and save frequencies (in repetitions)
    plot_freq = 100
    save_freq = 100

    last_radius = radius

    # take 80 O-D steps and then do mech eq.
    repetitions = 10000
    od_steps = 80
    for rep in range(1, repetitions+1):
        # make a number of O-D time steps
        od_start = time.time()
        for ts in range(1, od_steps+1):
            time_step(eta, eta_k)
        iterations += od_steps
        od_time = time.time() - od_start
        # Do elastic equilibrium
        meq_iter = mechanical_equilibrium(eta, eta_k)
        meq_time = time.time() - od_start - od_time
        energy = calculate_energy(eta, eta_k)
        radius = defect_radius(eta)
        total_time = time.time()-time_start
        stime = iterations*dt

        print("iter: %d; stime: %.3f; energy: %.16e; radius: %5.1f; od_time: %4.1f; meq_iter: %4d; meq_time: %.1f; total_time: %.1f" %
              (iterations, iterations*dt, energy, radius, od_time, meq_iter, meq_time, total_time))
        simulation_data.append([iterations, stime, energy, radius, meq_iter, total_time])

        if rep % plot_freq == 0 or last_radius < radius:
            update_plot(quad, stime, eta, datapath+"fig/", radius)
        if rep % save_freq == 0 or last_radius < radius:
            np.savez(datapath+"time%.0f"%stime, np.array(simulation_data), eta)

        if last_radius < radius:
            break
        last_radius = radius


def continue_calculation(path, datafile):

    # Load the data
    data = np.load(path+datafile)
    simulation_data = data['arr_0'].tolist()
    eta = data['arr_1']

    # Calculate some values that are the same for each time step
    # and should be calculated only once
    time_start = time.time()
    calculate_coefficients_tile()
    print("Coefficient calculation: %.3f s" % (time.time()-time_start))

    eta_k = np.fft.fft2(eta)
    run_calculation(simulation_data, eta, eta_k, path)


def plot_in_kspace(eta_k):
    plotting_signal_fft = np.fft.fftshift(np.real(eta_k[0]))
    plotting_k_x_values = np.fft.fftshift(k_x_values)
    plotting_k_y_values = np.fft.fftshift(k_y_values)
    min_x = np.min(plotting_k_x_values)
    max_x = np.max(plotting_k_x_values)
    min_y = np.min(plotting_k_y_values)
    max_y = np.max(plotting_k_y_values)
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(plotting_k_x_values, plotting_k_y_values, plotting_signal_fft)
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    plt.show()


def main():
    # Allocate memory for the complex amplitudes
    eta = np.zeros((3, nx, ny), dtype=np.complex128)
    
    # Initialize state to rotated grain and calculate Fourier transform
    init_state_seed(eta)

    fig = plt.figure(figsize=(10, 10))
    plt.pcolormesh(abs(eta[0]) + abs(eta[1]) + abs(eta[2]))
    plt.xlim([0, nx])
    plt.ylim([0, ny])
    plt.savefig("./fig/phi.png", dpi=200)

    # eta_k = np.fft.fft2(eta)
    #
    # # Calculate derivative operators in k-space
    # calculate_coefficients_tile()
    #
    # # Take 80 PFC time steps
    # for ts in range(100000):
    #     time_step(eta, eta_k)
    #     if ts % int(30/dt) == 0:
    #             fig = plt.figure(figsize=(10, 10))
    #             plt.pcolormesh(abs(eta[0])+abs(eta[1])+abs(eta[2]))
    #             plt.xlim([0, nx])
    #             plt.ylim([0, ny])
    #             plt.title("time %.1f" % (ts*dt))
    #             plt.savefig("./fig/seed_dt0.0125_%.0f.png" % (ts*dt), dpi=200)
    #             plt.close(fig)

    # # Run LBFGS algorithm for mechanical equilibration
    # equilibrium_algorithms_2.lbfgs_enh(
    #     eta, calculate_energy, calc_grad_theta, nx, ny)
    # eta_k[:] = np.fft.fft2(eta)

    # Save eta to file
    # np.save("./data/test_run/test", eta)


if __name__ == "__main__":
    main()
