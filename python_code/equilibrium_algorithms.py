
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time


def element_wise_norm(array, num_points):
    """
    :param array: an array of (3, nx, ny), where nx*ny=num_points
    :return: the norm
    """
    return np.linalg.norm(array.ravel(), ord=1)/(3*num_points)


def steepest_descent_fixed_dz(eta, calculate_energy, calc_grad_theta, nx, ny):
    """
    This is the easiest solution: uses a fixed time step to slowly work towards the minimum
    Very slow convergence
    """
    # fake time step; max number of iterations
    # check frequency - after how many iterations to check for energy and tolerance
    # tolerance - the goal error
    dz = 3.0
    max_iter = 200000
    check_freq = 100
    tolerance = 7.5e-9

    convergence_data = []
    num_fts = 0

    eta_k = np.fft.fft2(eta)
    last_it_energy = calculate_energy(eta, eta_k)
    num_fts += 2

    time_var = time.time()

    phi = np.absolute(eta)

    for it in range(1, max_iter+1):
        # in each iteration, decompose eta to theta and phi
        # and calculate the gradient of theta
        theta_old = np.angle(eta)
        grad_theta = calc_grad_theta(eta, eta_k)
        num_fts += 1

        theta_new = theta_old - dz*grad_theta

        eta[:] = phi * np.exp(1j*theta_new)
        eta_k[:] = np.fft.fft2(eta)
        num_fts += 1

        if it % check_freq == 0:
            energy = calculate_energy(eta, eta_k)
            num_fts += 1
            error = element_wise_norm(grad_theta, nx*ny)
            dur = time.time()-time_var
            time_var = time.time()
            print("it %6d: energy: %.16e; err: %.16e; fts: %d; time: %5.1f" % (it, energy, error, num_fts, dur))

            convergence_data.append([it, energy, error, num_fts, dur])
            if energy > last_it_energy:
                print("Error: the energy increased by %.16e" % (energy-last_it_energy))
            last_it_energy = energy
            if error < tolerance:
                print("Solution found.")
                break
        if it == max_iter:
            print("Solution was not found within %d iterations, returning un-converged state." % it)

    #np.savez("./data/mech_run/sd_fixed_dz", np.array(convergence_data), eta, eta)


def steepest_descent_fixed_dz_finite_diff(eta, calculate_energy_an, calc_grad_theta_an, calculate_energy_fd, calc_grad_theta_fd, nx, ny):
    """
    This is the easiest solution: uses a fixed time step to slowly work towards the minimum
    Very slow convergence
    """
    # check frequency - after how many iterations to check for energy and tolerance
    # tolerance - the goal error
    dz = 1.0
    max_iter = 200000
    check_freq = 100
    tolerance = 7.5e-9

    convergence_data = []
    num_fts = 0

    eta_k = np.fft.fft2(eta)
    last_it_energy = calculate_energy_fd(eta, eta_k)
    num_fts += 2

    time_var = time.time()

    phi = np.absolute(eta)

    for it in range(1, max_iter+1):
        # in each iteration, decompose eta to theta and phi
        # and calculate the gradient of theta
        theta_old = np.angle(eta)
        grad_theta = calc_grad_theta_fd(eta, eta_k)

        theta_new = theta_old - dz*grad_theta

        eta[:] = phi * np.exp(1j*theta_new)
        eta_k[:] = np.fft.fft2(eta)
        num_fts += 1

        if it % check_freq == 0:
            energy = calculate_energy_fd(eta, eta_k)
            num_fts += 1
            energy_an = calculate_energy_an(eta, eta_k)
            grad_an = calc_grad_theta_an(eta, eta_k)
            error_an = element_wise_norm(grad_an, nx*ny)
            error = element_wise_norm(grad_theta, nx*ny)
            dur = time.time()-time_var
            time_var = time.time()
            print("it %6d: energy: %.16e; en_an: %.16e; err: %.16e; err_an: %.16e; fts: %d; time: %5.1f"
                  % (it, energy, energy_an, error, error_an, num_fts, dur))

            convergence_data.append([it, energy, error, num_fts, dur, energy_an, error_an])
            if energy > last_it_energy:
                print("Error: the energy increased by %.16e" % (energy-last_it_energy))
            last_it_energy = energy
            if error < tolerance:
                #print("Solution found.")
                break
        if it == max_iter:
            print("Solution was not found within %d iterations, returning un-converged state." % it)

    np.savez("./data/mech_run/sd_fixed_dz_fd", np.array(convergence_data), eta, eta)


def steepest_descent_exp_search(eta, calculate_energy, calc_grad_theta, num_points):
    """
    In each iteration, this solution tries to take the longest (but inaccurate) step towards the minimum
    """
    tolerance = 5.0e-9

    eta_save_tol = 7.5e-9
    extra_eta_saved = False

    time_var = time.time()

    convergence_data = []
    num_fts = 0

    eta_k = np.fft.fft2(eta)
    last_it_energy = calculate_energy(eta, eta_k)
    grad_theta = calc_grad_theta(eta, eta_k)
    num_fts += 3

    phi = np.absolute(eta)

    max_iter = 200000
    for it in range(1, max_iter+1):
        theta_old = np.angle(eta)
        dz, energy, eta[:], nft = exp_line_search(last_it_energy, theta_old, phi,
                                                    - grad_theta, calculate_energy)
        num_fts += nft

        eta_k[:] = np.fft.fft2(eta)
        num_fts += 1
        last_it_energy = energy

        # estimate error
        grad_theta[:] = calc_grad_theta(eta, eta_k)
        num_fts += 1
        error = element_wise_norm(grad_theta, num_points)

        dur = time.time()-time_var
        time_var = time.time()
        print("it %5d: took a step of %5.2f; energy: %.16e; err: %.16e; fts: %d; time: %.1f"
              % (it, dz, energy, error, num_fts, dur))
        convergence_data.append([it, energy, error, num_fts, dur, dz])
        if error < eta_save_tol and not extra_eta_saved:
            eta_extra = np.copy(eta)
            extra_eta_saved = True
        if error < tolerance:
            print("Goal tolerance reached, solution found.")
            break

        if it == max_iter:
            print("Solution was not found within %d iterations, returning un-converged state." % it)

    #np.savez("./data/mech_run/sd_exp_search", np.array(convergence_data), eta, eta_extra)


def accelerated_steepest_descent(eta, calculate_energy, calc_grad_theta, num_points):
    """
    Nesterov accelerated gradient descent with "fixed step" (in reality, it acquires a high terminal velocity)
    """

    dz = 1.0
    check_freq = 100
    tolerance = 7.5e-9

    #gamma = 0.997
    gamma = 0.995
    last_velocity = 0

    convergence_data = []
    num_fts = 0

    time_var = time.time()

    eta_k = np.fft.fft2(eta)
    last_it_energy = calculate_energy(eta, eta_k)
    num_fts += 2

    phi = np.absolute(eta)

    max_iter = 10000
    for it in range(1, max_iter+1):
        # in each iteration, decompose eta to theta and phi
        # and calculate the gradient of theta
        theta_old = np.angle(eta)

        theta_predict = theta_old + gamma*last_velocity

        eta_predict = phi * np.exp(1j*theta_predict)
        eta_k_predict = np.fft.fft2(eta_predict)
        num_fts += 1

        grad_theta = calc_grad_theta(eta_predict, eta_k_predict)
        num_fts += 1

        velocity = gamma*last_velocity - dz * grad_theta

        theta_new = theta_old + velocity

        eta[:] = phi * np.exp(1j*theta_new)
        eta_k[:] = np.fft.fft2(eta)
        num_fts += 1

        last_velocity = velocity

        if it % check_freq == 0:
            energy = calculate_energy(eta, eta_k)
            num_fts += 1
            error = element_wise_norm(grad_theta, num_points)
            error1 = np.linalg.norm(grad_theta.ravel())
            error2 = np.linalg.norm(grad_theta.ravel(), ord=np.inf)

            dur = time.time() - time_var
            time_var = time.time()
            print("it %5d: energy: %.16e; err: %.16e; err1: %.16e; err2: %.16e; fts: %d; time: %.1f" % (it, energy, error, error1, error2, num_fts, dur))
            convergence_data.append([it, energy, error, error1, error2, num_fts, dur])
            if energy > last_it_energy:
                print("Error: the energy increased by %.16e" % (energy-last_it_energy))
            last_it_energy = energy
            if error < tolerance:
                print("Solution found.")
                break
        if it == max_iter:
            print("Solution was not found within %d iterations, returning un-converged state." % it)

    np.save("./data/mech_run_big/accel_sd", np.array(convergence_data))


def accelerated_steepest_descent_modular(eta, eta_k, calc_grad_theta, num_points, tolerance, max_iter):
    """
    Nesterov accelerated gradient descent to be called from other algorithms (lbfgs)
    """

    dz = 1.0
    check_freq = 100

    gamma = 0.992
    last_velocity = 0

    num_fts = 0

    time_var = time.time()

    phi = np.absolute(eta)

    #max_iter = 400
    for it in range(1, max_iter+1):

        theta_old = np.angle(eta)

        theta_predict = theta_old + gamma*last_velocity

        eta_predict = phi * np.exp(1j*theta_predict)
        eta_k_predict = np.fft.fft2(eta_predict)
        num_fts += 1

        grad_theta = calc_grad_theta(eta_predict, eta_k_predict)
        num_fts += 1

        velocity = gamma*last_velocity - dz * grad_theta

        theta_new = theta_old + velocity

        eta[:] = phi * np.exp(1j*theta_new)
        eta_k[:] = np.fft.fft2(eta)
        num_fts += 1

        last_velocity = velocity

        if it % check_freq == 0:
            error = element_wise_norm(grad_theta, num_points)
            if error < tolerance:
                break

    return error, num_fts



def accelerated_steepest_descent_adaptive_dz(eta, calculate_energy, calc_grad_theta, num_points):
    """
    Nesterov accelerated gradient descent with adaptive time step
    """
    dz_ad = 1.0
    max_iter = 100000
    tolerance = 7.5e-9

    adaptive_step_freq = 100
    num_adaptive_steps = 5

    check_freq = 50

    gamma = 0.9
    last_velocity = 0

    convergence_data = []
    num_fts = 0

    eta_save_tol = 7.5e-9
    extra_eta_saved = False

    time_var = time.time()

    eta_k = np.fft.fft2(eta)
    last_it_energy = calculate_energy(eta, eta_k)
    num_fts += 2

    phi = np.absolute(eta)

    it = 1
    while it <= max_iter:

        if (it-1) % adaptive_step_freq == 0:
            # ------------------------------------------------------------------------
            # run the specified number of adaptive (no acceleration) time steps
            for sn in range(num_adaptive_steps):
                theta_old = np.angle(eta)
                phi = np.absolute(eta)
                grad_theta = calc_grad_theta(eta, eta_k)
                num_fts += 1
                #dz, energy, eta[:], nft = exp_line_search_with_interpolation(last_it_energy, theta_old, phi,
                #                                                             - grad_theta, calculate_energy)
                dz, energy, eta[:], nft = exp_line_search(last_it_energy, theta_old, phi,
                                                                             - grad_theta, calculate_energy)
                num_fts += nft
                eta_k[:] = np.fft.fft2(eta)
                num_fts += 1
                last_it_energy = energy
                error = element_wise_norm(grad_theta, num_points)
                dur = time.time() - time_var
                time_var = time.time()
                print("it %5d: adaptive step: %8.1f; energy: %.16e; err: %.16e; fts: %d; time: %.2f" %
                      (it, dz, energy, error, num_fts, dur))
                convergence_data.append([it, energy, error, num_fts, dur, dz])
                if error < tolerance:
                    break
                it += 1
            last_velocity = 0.0
            # ------------------------------------------------------------------------

        # Resume with accelerated descent
        # in each iteration, decompose eta to theta and phi
        # and calculate the gradient of theta
        theta_old = np.angle(eta)

        theta_predict = theta_old + gamma*last_velocity

        eta_predict = phi * np.exp(1j*theta_predict)
        eta_k_predict = np.fft.fft2(eta_predict)
        num_fts += 1

        grad_theta = calc_grad_theta(eta_predict, eta_k_predict)
        num_fts += 1

        velocity = gamma*last_velocity - dz_ad * grad_theta

        theta_new = theta_old + velocity

        eta[:] = phi * np.exp(1j*theta_new)
        eta_k[:] = np.fft.fft2(eta)
        num_fts += 1

        last_velocity = velocity

        if it % check_freq == 0:
            energy = calculate_energy(eta, eta_k)
            num_fts += 1
            error = element_wise_norm(grad_theta, num_points)
            dur = time.time() - time_var
            time_var = time.time()
            print("it %5d: energy: %.16e; err: %.16e; fts: %d; time: %.1f" % (it, energy, error, num_fts, dur))
            convergence_data.append([it, energy, error, num_fts, dur, dz_ad])
            if energy > last_it_energy:
                print("Error: the energy increased by %.16e" % (energy-last_it_energy))
            last_it_energy = energy
            if error < eta_save_tol and not extra_eta_saved:
                eta_extra = np.copy(eta)
                extra_eta_saved = True
            if error < tolerance:
                break
        if it >= max_iter:
            print("Solution was not found within %d iterations, returning un-converged state." % it)
        it += 1

    #np.savez("./data/mech_run/acc_sd_adz", np.array(convergence_data), eta, eta_extra)
    return it


def accelerated_steepest_descent_adaptive_dz_finite_diff(eta, calculate_energy, calc_grad_theta_fd, num_points):
    """
    Nesterov accelerated gradient descent with adaptive time step
    """
    dz_ad = 1.0
    max_iter = 100000
    tolerance = 7.5e-9

    adaptive_step_freq = 100
    num_adaptive_steps = 4

    check_freq = 50

    gamma = 0.9
    last_velocity = 0

    convergence_data = []
    num_fts = 0

    eta_save_tol = 7.5e-9
    extra_eta_saved = False

    time_var = time.time()

    eta_k = np.fft.fft2(eta)
    last_it_energy = calculate_energy(eta, eta_k)
    num_fts += 2

    phi = np.absolute(eta)

    it = 1
    while it <= max_iter:

        if (it-1) % adaptive_step_freq == 0:
            # ------------------------------------------------------------------------
            # run the specified number of adaptive (no acceleration) time steps
            for sn in range(num_adaptive_steps):
                theta_old = np.angle(eta)
                phi = np.absolute(eta)
                grad_theta = calc_grad_theta_fd(eta)

                dz, energy, eta[:], nft = exp_line_search(last_it_energy, theta_old, phi,
                                                                             - grad_theta, calculate_energy)
                num_fts += nft
                eta_k[:] = np.fft.fft2(eta)
                num_fts += 1
                last_it_energy = energy
                error = element_wise_norm(grad_theta, num_points)
                dur = time.time() - time_var
                time_var = time.time()
                print("it %5d: adaptive step: %8.1f; energy: %.16e; err: %.16e; fts: %d; time: %.2f" %
                      (it, dz, energy, error, num_fts, dur))
                convergence_data.append([it, energy, error, num_fts, dur, dz])
                if error < tolerance:
                    break
                it += 1
            last_velocity = 0.0
            # ------------------------------------------------------------------------

        # Resume with accelerated descent
        # in each iteration, decompose eta to theta and phi
        # and calculate the gradient of theta
        theta_old = np.angle(eta)

        theta_predict = theta_old + gamma*last_velocity

        eta_predict = phi * np.exp(1j*theta_predict)

        grad_theta = calc_grad_theta_fd(eta_predict)

        velocity = gamma*last_velocity - dz_ad * grad_theta

        theta_new = theta_old + velocity

        eta[:] = phi * np.exp(1j*theta_new)

        last_velocity = velocity

        if it % check_freq == 0:
            eta_k[:] = np.fft.fft2(eta)
            num_fts += 1
            energy = calculate_energy(eta, eta_k)
            num_fts += 1
            error = element_wise_norm(grad_theta, num_points)
            dur = time.time() - time_var
            time_var = time.time()
            print("it %5d: energy: %.16e; err: %.16e; fts: %d; time: %.1f" % (it, energy, error, num_fts, dur))
            convergence_data.append([it, energy, error, num_fts, dur, dz_ad])
            if energy > last_it_energy:
                print("Error: the energy increased by %.16e" % (energy-last_it_energy))
            last_it_energy = energy
            if error < eta_save_tol and not extra_eta_saved:
                eta_extra = np.copy(eta)
                extra_eta_saved = True
            if error < tolerance:
                break
        if it >= max_iter:
            print("Solution was not found within %d iterations, returning un-converged state." % it)
        it += 1

    #np.savez("./data/mech_run/acc_sd_adz", np.array(convergence_data), eta, eta_extra)
    #return it

def bisection_search(x_a, f_a, x_b, f_b, func, tol):
    x_m = 1/2*(x_a+x_b)
    f_m = func(x_m)[0]
    x = np.array([x_a, 0, x_m, 0, x_b])
    f = np.array([f_a, 0, f_m, 0, f_b])
    for i in range(100):
        x[1] = 1/2*(x[0]+x[2])
        x[3] = 1/2*(x[2]+x[4])
        f[1] = func(x[1])[0]
        f[3] = func(x[3])[0]
        min_ind = f.argmin()
        if min_ind == 0 or min_ind == 1:
            x[4], f[4] = x[2], f[2]
            x[2], f[2] = x[1], f[1]
        elif min_ind == 2:
            x[4], f[4] = x[3], f[3]
            x[0], f[0] = x[1], f[1]
        else:
            x[0], f[0] = x[2], f[2]
            x[2], f[2] = x[3], f[3]
        if np.abs(x[4]-x[0]) < tol:
            break
    min_ind = f.argmin()
    return x[min_ind], f[min_ind]


def take_step(dz, theta, phi, grad_theta, calculate_energy):
    theta_new = theta - dz*grad_theta
    eta_new = phi * np.exp(1j*theta_new)
    eta_k_new = np.fft.fft2(eta_new)
    energy = calculate_energy(eta_new, eta_k_new)
    return energy, eta_new, eta_k_new


def steepest_descent_bin_search(eta, calculate_energy, calc_grad_theta, num_points):
    """
    In each iteration, this algorithm uses bisection search to find the time step that minimizes the energy
    Definitely inferior to the previous method; slow performance and convergence
    """
    dz_start = 1.0
    search_factor = 2.0
    max_iter = 10000
    tolerance = 7.5e-9

    last_it_energy = calculate_energy(eta)
    eta_k = np.fft.fft2(eta)

    for it in range(1, max_iter+1):
        # in each iteration, decompose eta to theta and phi
        # and calculate the gradient of theta
        theta_old = np.angle(eta)
        phi = np.absolute(eta)
        grad_theta = calc_grad_theta(eta, eta_k)

        error = element_wise_norm(grad_theta, num_points)
        if error < tolerance:
            print("Solution found. Final error: %.16e" % error)
            break

        # First, increase time step exponentially until the minimum is crossed
        dz_queue = deque([0.0, 0.0, 0.0])
        en_queue = deque([last_it_energy]*3)

        dz = dz_start/search_factor
        for i in range(1, 100):
            dz *= search_factor
            theta_probe = theta_old - dz*grad_theta
            eta_probe = phi * np.exp(1j*theta_probe)
            energy_probe = calculate_energy(eta_probe)

            dz_queue.append(dz)
            dz_queue.popleft()
            en_queue.append(energy_probe)
            en_queue.popleft()

            if energy_probe > en_queue[-2]:
                break

        func = lambda x: take_step(x, theta_old, phi, grad_theta, calculate_energy)
        dz, en = bisection_search(dz_queue[0], en_queue[0], dz_queue[2], en_queue[2], func, 0.01)

        # finally, take the step

        theta_new = theta_old - dz*grad_theta
        eta[:] = phi * np.exp(1j*theta_new)
        eta_k[:] = np.fft.fft2(eta)

        print("it %5d: took a step of %5.2f; energy: %.16e; err: %.16e" %
                      (it, dz, en, error))

        if it == max_iter:
            print("Solution was not found within %d iterations, returning un-converged state." % it)


def exp_line_search(energy_start, theta_old, phi, direction, calculate_energy):
    """
    Multiplies (or divides) the step by search_factor until a suitable step is found
    """
    dz_start = 1.0
    search_factor = 2.0

    plot_data = []
    num_fts = 0  # number of fourier transforms performed

    def make_plot():
        np_p_d = np.array(plot_data)
        plt.plot(np_p_d[:,0], np_p_d[:,1], 'bo')
        plt.show()


    theta_new = theta_old + dz_start*direction
    eta_new = phi * np.exp(1j*theta_new)
    eta_k_new = np.fft.fft2(eta_new)
    energy_new = calculate_energy(eta_new, eta_k_new)
    num_fts += 2

    # if the first step energy is higher than starting, decrease step size
    if energy_new > energy_start:
        search_factor = 1.0/search_factor

    eta_last = eta_new
    energy_last = energy_new

    plot_data.append([0, energy_start])
    plot_data.append([dz_start, energy_new])
    #print("dz: %5.1f; en: %.16e"%(dz_start, energy_new))

    dz = dz_start
    for i in range(20):
        # increase (or decrease) the step size
        dz *= search_factor

        # take the step and calculate the resulting energy
        theta_new = theta_old + dz*direction
        eta_new = phi * np.exp(1j*theta_new)
        eta_k_new = np.fft.fft2(eta_new)
        energy_new = calculate_energy(eta_new, eta_k_new)
        num_fts += 2
        #print("dz: %5.1f; en: %.16e"%(dz, energy_new))

        plot_data.append([dz, energy_new])

        # in case of increasing step size, if new energy is bigger, this is the first unsuitable step
        #  and return last step results
        if energy_new > energy_last and search_factor > 1:
            #make_plot()
            return dz/search_factor, energy_last, eta_last, num_fts
        # in case of decreasing step size, if new energy is smaller than the starting energy,
        # then this is the first suitable step and return current step results
        if energy_new < energy_start and search_factor < 1:
            #make_plot()
            return dz, energy_new, eta_new, num_fts

        eta_last = eta_new
        energy_last = energy_new

        if search_factor < 1 and i > 7:
            break

    print("Didn't find step, taking best step.")
    #make_plot()
    return dz, energy_new, eta_new, num_fts


def quadratic_interpolation_minimum(x, y):
    x_mat = np.empty((3,3))
    x_mat[:, 0] = x**2
    x_mat[:, 1] = x
    x_mat[:, 2] = 1
    a = np.dot(np.linalg.inv(x_mat), y)
    x_min = -a[1]/(2*a[0])
    return x_min


def exp_line_search_with_interpolation(energy_start, theta_old, phi, direction, calculate_energy):
    """
    Multiplies (or divides) the step by search_factor until a suitable step is found
    and additionally uses quadratic interpolation to get a more accurate result
    """
    dz_start = 1.0
    search_factor = 2.0

    plot_data = []
    num_fts = 2  # number of fourier transforms performed

    def make_plot():
        np_p_d = np.array(plot_data)
        plt.plot(np_p_d[:,0], np_p_d[:,1], 'bo')
        plt.plot(dz_best, energy_best, 'go')
        plt.show()

    theta_new = theta_old + dz_start*direction
    eta_new = phi * np.exp(1j*theta_new)
    eta_k_new = np.fft.fft2(eta_new)
    energy_new = calculate_energy(eta_new, eta_k_new)

    # if the first step energy is higher than starting, decrease step size
    if energy_new > energy_start:
        search_factor = 1.0/search_factor

    eta_last = eta_new
    energy_last = energy_new

    plot_data.append([0, energy_start])
    plot_data.append([dz_start, energy_new])

    dz_queue = deque([0.0, 0.0, dz_start])
    en_queue = deque([energy_start, energy_start, energy_new])

    dz = dz_start
    for i in range(20):
        # increase (or decrease) the step size
        dz *= search_factor

        # take the step and calculate the resulting energy
        theta_new = theta_old + dz*direction
        eta_new = phi * np.exp(1j*theta_new)
        eta_k_new = np.fft.fft2(eta_new)
        energy_new = calculate_energy(eta_new, eta_k_new)
        num_fts += 2

        plot_data.append([dz, energy_new])
        dz_queue.append(dz)
        dz_queue.popleft()
        en_queue.append(energy_new)
        en_queue.popleft()

        # in case of increasing step size, if new energy is bigger, this is the first unsuitable step
        if energy_new > energy_last and search_factor > 1:
            dz_best = quadratic_interpolation_minimum(np.array(dz_queue), np.array(en_queue))
            theta_best = theta_old + dz_best*direction
            eta_best = phi * np.exp(1j*theta_best)
            eta_k_best = np.fft.fft2(eta_best)
            energy_best = calculate_energy(eta_best, eta_k_best)
            num_fts += 2
            if energy_best > en_queue[-1]:
                print("Quadratic min higher than %.16f" % (energy_best-en_queue[-1]))
            #make_plot()
            return dz_best, energy_best, eta_best, num_fts

        # in case of decreasing step size, if new energy is smaller than the starting energy,
        # then this is the first suitable step and return current step results
        if energy_new < energy_start and search_factor < 1:
            dz_queue.append(0.0)
            dz_queue.popleft()
            en_queue.append(energy_start)
            en_queue.popleft()
            dz_best = quadratic_interpolation_minimum(np.array(dz_queue), np.array(en_queue))
            theta_best = theta_old + dz_best*direction
            eta_best = phi * np.exp(1j*theta_best)
            eta_k_best = np.fft.fft2(eta_best)
            energy_best = calculate_energy(eta_best, eta_k_best)
            num_fts += 2
            #make_plot()
            return dz_best, energy_best, eta_best, num_fts

        eta_last = eta_new
        energy_last = energy_new

        if search_factor < 1 and i > 7:
            break

    print("Didn't find step, returning initial state.")
    #make_plot()
    return 0, energy_start, phi * np.exp(1j*theta_old), num_fts


def conjugate_gradient(eta, calculate_energy, calc_grad_theta, nx, ny):

    tolerance = 7.5e-9

    convergence_data = []
    num_fts = 0

    eta_k = np.fft.fft2(eta)
    energy_start = calculate_energy(eta, eta_k)
    num_fts += 2

    # take the first step (steepest descent direction)
    theta_old = np.angle(eta)
    phi = np.absolute(eta)
    grad_theta = calc_grad_theta(eta, eta_k)
    num_fts += 1

    # perform a (loose) line search
    dz, energy, eta[:], nft = exp_line_search_with_interpolation(energy_start, theta_old, phi, -grad_theta, calculate_energy)
    num_fts += nft

    # update stuff
    eta_k[:] = np.fft.fft2(eta)
    num_fts += 1
    last_grad = np.copy(grad_theta)
    last_energy = energy

    # estimate error (and calculate gradient for the next step)
    grad_theta[:] = calc_grad_theta(eta, eta_k)
    num_fts += 2
    error = element_wise_norm(grad_theta, nx*ny)
    print("it %5d: took a step of %5.2f; energy: %.16e; err: %.16e" % (0, dz, energy, error))
    convergence_data.append([0, dz, energy, error, num_fts])

    # now, start taking conjugate steps
    conjugate_direction = - last_grad
    max_iter = 10000
    for it in range(1, max_iter+1):
        # 1) gradient is known from last step (from error estimation)
        # 2) compute beta (Polak-Ribiere)
        beta = np.zeros(3)
        for j in range(3):
            beta[j] = (np.dot(grad_theta[j].ravel(), (grad_theta[j]-last_grad[j]).ravel()) /
                       np.dot(last_grad[j].ravel(), last_grad[j].ravel()))
            if beta[j] < 0:
                print("Reset for %d!" % j)
            beta[j] = max(0, beta[j])
        # 3) update conjugate direction
        conjugate_direction = - grad_theta + np.array([beta[j]*conjugate_direction[j] for j in range(3)])
        # 4) perform a (loose) line search and 5) update position
        theta_old = np.angle(eta)
        phi = np.absolute(eta)
        dz, energy, eta[:], nft = exp_line_search_with_interpolation(last_energy, theta_old, phi, conjugate_direction, calculate_energy)
        num_fts += nft

        eta_k[:] = np.fft.fft2(eta)
        num_fts += 1
        last_grad[:] = grad_theta
        last_energy = energy

        # 6) estimate error
        grad_theta[:] = calc_grad_theta(eta, eta_k)
        num_fts += 2
        error = element_wise_norm(grad_theta, nx*ny)
        print("it %5d: took a step of %5.2f; energy: %.16e; err: %.16e" % (it, dz, energy, error))
        convergence_data.append([it, dz, energy, error, num_fts])
        if error < tolerance:
            print("Goal tolerance reached, solution found.")
            break

        if it == max_iter:
            print("Solution was not found within %d iterations, returning un-converged state." % it)

    np.save("./data/cg_interp_%d" % it, np.array(convergence_data))



def conjugate_gradient_fixed_dz(eta, calculate_energy, calc_grad_theta, nx, ny):

    dz = 3.0
    tolerance = 7.5e-9
    check_freq = 100

    convergence_data = []
    num_fts = 0

    time_var = time.time()

    eta_k = np.fft.fft2(eta)
    energy_start = calculate_energy(eta, eta_k)
    num_fts += 2

    # take the first step (steepest descent direction)
    theta_old = np.angle(eta)
    phi = np.absolute(eta)
    grad_theta = calc_grad_theta(eta, eta_k)
    num_fts += 1

    theta_new = theta_old - dz*grad_theta

    # update stuff
    eta[:] = phi * np.exp(1j*theta_new)
    eta_k[:] = np.fft.fft2(eta)
    last_grad = np.copy(grad_theta)
    energy = calculate_energy(eta, eta_k)
    num_fts += 2

    # estimate error (and calculate gradient for the next step)
    grad_theta[:] = calc_grad_theta(eta, eta_k)
    num_fts += 1
    error = element_wise_norm(grad_theta, nx*ny)
    #print("it %5d; step: %2.1f; energy: %.16e; err: %.16e" % (0, dz, energy, error))
    #convergence_data.append([0, dz, energy, error, num_fts])

    last_it_energy = energy

    # now, start taking conjugate steps
    conjugate_direction = - last_grad
    max_iter = 200000
    for it in range(1, max_iter+1):
        # 1) gradient is known from last step (from error estimation)
        # 2) compute beta (Polak-Ribiere)
        beta = (np.dot(grad_theta.ravel(), (grad_theta-last_grad).ravel()) /
                np.dot(last_grad.ravel(), last_grad.ravel()))
        beta = max(0.0, beta)
        # 3) update conjugate direction
        conjugate_direction = - grad_theta + beta*conjugate_direction
        # 4) take the step
        theta_old = np.angle(eta)
        phi = np.absolute(eta)
        theta_new = theta_old + dz*conjugate_direction

        eta[:] = phi * np.exp(1j*theta_new)
        eta_k[:] = np.fft.fft2(eta)
        num_fts += 1
        last_grad[:] = grad_theta

        # 6) calculate gradient for next step
        grad_theta[:] = calc_grad_theta(eta, eta_k)
        num_fts += 1

        # 7) calculate energy and error
        if it % check_freq == 0:
            error = element_wise_norm(grad_theta, nx*ny)
            energy = calculate_energy(eta, eta_k)
            num_fts += 1
            dur = time.time()-time_var
            time_var = time.time()
            print("it %5d: step: %2.1f; energy: %.16e; err: %.16e; fts: %7d; time: %.1f" %
                  (it, dz, energy, error, num_fts, dur))
            convergence_data.append([it, energy, error, num_fts, dur])
            if energy > last_it_energy:
                print("Error: the energy increased by %.16e" % (energy-last_it_energy))
            last_it_energy = energy
            if error < tolerance:
                print("Goal tolerance reached, solution found.")
                break

        if it == max_iter:
            print("Solution was not found within %d iterations, returning un-converged state." % it)

    np.savez("./data/mech_run/cg_fixed", np.array(convergence_data), eta, eta)