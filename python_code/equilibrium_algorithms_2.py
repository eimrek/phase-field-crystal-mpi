
import numpy as np
import matplotlib.pyplot as plt
import time
import equilibrium_algorithms
from collections import deque


def element_wise_norm(array, num_points):
    """
    :param array: an array of (3, nx, ny), where nx*ny=num_points
    :return: the norm
    """
    return np.linalg.norm(array.ravel(), ord=1)/(3*num_points)


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
    for i in range(100):
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


def lbfgs_direction(s, y, grad):
    m = len(s)
    q = np.copy(grad)
    alpha = []
    rho = np.empty(m)

    for i in range(m):
        rho[i] = 1.0/np.dot(s[i], y[i])

    for i in range(m):
        alpha_i = rho[i]*np.dot(s[i], q)
        alpha.append(alpha_i)
        q = q - alpha_i*y[i]

    r = q  # H_-m = identity

    for i in range(m-1, -1, -1):
        beta = rho[i]*np.dot(y[i], r)
        r = r + s[i]*(alpha[i]-beta)
    return r


def lbfgs(eta, calculate_energy, calc_grad_theta, nx, ny):

    tolerance = 7.5e-9
    print_freq = 100
    dz = 0.01

    eta_save_tol = 7.5e-9
    eta_extra_saved = False

    time_var = time.time()

    # Calculate phi at the start, as it will remain constant
    phi = np.absolute(eta)

    eta_k = np.fft.fft2(eta)
    num_fts = 1

    # Starting values
    g_0 = calc_grad_theta(eta, eta_k)
    num_fts += 1
    theta_0 = np.angle(eta)

    s = []
    y = []

    m = 0
    max_m = 5

    g_last = g_0
    theta_last = theta_0

    convergence_data = []

    for it in range(10000):
        r = lbfgs_direction(s, y, g_last.flatten()).reshape(eta.shape)

        theta_new = theta_last - dz*r
        eta_new = phi*np.exp(1j*theta_new)
        eta_k_new = np.fft.fft2(eta_new)
        num_fts += 1
        g_new = calc_grad_theta(eta_new, eta_k_new)
        num_fts += 1

        s.append((theta_new-theta_last).flatten())
        y.append((g_new - g_last).flatten())

        if m < max_m:
            m += 1
        else:
            s.pop(0)
            y.pop(0)

        error = element_wise_norm(g_new, nx*ny)

        if it % print_freq == 0 or error < tolerance:
            en_new = calculate_energy(eta_new, eta_k_new)
            dur = time.time()-time_var
            time_var = time.time()
            print("it: %d; en: %.16e; err: %.16e; fts: %d; time: %.1f" % (it, en_new, error, num_fts, dur))
            convergence_data.append([it, en_new, error, num_fts, dur])

        if error < eta_save_tol and not eta_extra_saved:
            eta_extra = np.copy(eta_new)
            eta_extra_saved = True
        if error < tolerance:
            break

        g_last = g_new
        theta_last = theta_new

    eta[:] = eta_new
    #np.savez("./data/mech_run/lbfgs", np.array(convergence_data), eta, eta_extra)


def lbfgs_enh(eta, calculate_energy, calc_grad_theta, nx, ny):

    tolerance = 7.5e-9
    print_freq = 100
    dz = 1.0

    # Imitate a static variable in python
    if not hasattr(lbfgs_enh, "lbfgs_iter"):
        lbfgs_enh.lbfgs_iter = 2500

    lbfgs_iter_increment = 500
    max_acc_sd_iter = 400

    max_m = 5

    time_var = time.time()

    # Calculate phi at the start, as it will remain constant
    phi = np.absolute(eta)

    eta_ = eta
    eta_k = np.fft.fft2(eta_)
    num_fts = 1

    convergence_data = []
    total_lbfgs_iter = 0

    energy_0 = calculate_energy(eta_, eta_k)
    g_0 = calc_grad_theta(eta_, eta_k)

    # Main loop: 1) Line search; 2) lbfgs steps; 3) accelerated sd for error reduction
    for it in range(1000):

        # 1) Line search
        energy_0 = calculate_energy(eta_, eta_k)
        g_0 = calc_grad_theta(eta_, eta_k)
        num_fts += 1
        theta_0 = np.angle(eta_)
        ls_dz, en, eta_, fts = exp_line_search(energy_0, theta_0, phi, -g_0, calculate_energy)
        theta_last = np.angle(eta_)
        num_fts += fts
        print("Line search step: %.1f" % ls_dz)
        eta_k = np.fft.fft2(eta_)
        g_last = calc_grad_theta(eta_, eta_k)
        num_fts += 2

        s = []
        y = []

        # 2) L-BFGS steps
        current_lbfgs_iter = lbfgs_enh.lbfgs_iter
        if it != 0:
            current_lbfgs_iter = lbfgs_iter_increment
        for it_lbfgs in range(1, current_lbfgs_iter+1):
            r = lbfgs_direction(s, y, g_last.flatten()).reshape(eta_.shape)
            theta_new = theta_last - dz*r
            eta_ = phi*np.exp(1j*theta_new)
            eta_k = np.fft.fft2(eta_)
            g_new = calc_grad_theta(eta_, eta_k)
            num_fts += 2

            s.append((theta_new-theta_last).flatten())
            y.append((g_new - g_last).flatten())

            if len(s) >= max_m:
                s.pop(0)
                y.pop(0)

            g_last = g_new
            theta_last = theta_new

            if it_lbfgs % print_freq == 0:
                en_new = calculate_energy(eta_, eta_k)
                num_fts += 1
                error = element_wise_norm(g_new, nx*ny)
                dur = time.time()-time_var
                time_var = time.time()
                print("it: %d; en: %.16e; err: %.16e; fts: %d; time: %.1f" % (it_lbfgs, en_new, error, num_fts, dur))
                convergence_data.append([it, en_new, error, num_fts, dur])

                if error < tolerance:
                    break
        total_lbfgs_iter += it_lbfgs
        if error < tolerance:
            break

        # 3) Accelerated sd for error reduction
        err, fts = equilibrium_algorithms.accelerated_steepest_descent_modular(eta_, eta_k, calc_grad_theta, nx*ny, tolerance, max_acc_sd_iter)
        num_fts += fts
        energy = calculate_energy(eta_, eta_k)
        num_fts += 1
        dur = time.time()-time_var
        time_var = time.time()
        print("Acc. sd; en: %.16e; err: %.16e; fts: %d; time: %.1f" % (energy, err, num_fts, dur))
        convergence_data.append([it, en_new, err, num_fts, dur])
        if err < tolerance:
            break

    eta[:] = eta_
    np.save("./data/mech_run_big/lbfgs_enh", np.array(convergence_data))
    lbfgs_enh.lbfgs_iter = total_lbfgs_iter
    return total_lbfgs_iter


def adagrad(eta, calculate_energy, calc_grad_theta, nx, ny):

    tolerance = 7.5e-9
    print_freq = 100
    dz = 1.0

    time_var = time.time()

    # Calculate phi at the start, as it will remain constant
    phi = np.absolute(eta)

    eta_k = np.fft.fft2(eta)
    num_fts = 1

    # Starting values
    theta_last = np.angle(eta)
    g_last = calc_grad_theta(eta, eta_k)
    num_fts += 1

    sum_grad_sqs = g_last**2
    eps = 1e-8

    convergence_data = []

    max_it = 10000
    for it in range(1, max_it+1):
        if it == 0:
            theta_new = theta_last - dz*g_last
        else:
            theta_new = theta_last - dz*g_last/np.sqrt(sum_grad_sqs+eps)

        eta_new = phi*np.exp(1j*theta_new)
        eta_k_new = np.fft.fft2(eta_new)
        num_fts += 1
        g_new = calc_grad_theta(eta_new, eta_k_new)
        num_fts += 1

        sum_grad_sqs += g_last**2

        error = element_wise_norm(g_new, nx*ny)

        if it%print_freq == 0 or error < tolerance:
            energy = calculate_energy(eta_new, eta_k_new)
            dur = time.time()-time_var
            time_var = time.time()
            print("it: %d; en: %.16e; err: %.16e; fts: %d; time: %.1f" % (it, energy, error, num_fts, dur))
            convergence_data.append([it, energy, error, num_fts, dur])

        if error < tolerance:
            break

        theta_last = theta_new
        g_last = g_new

    eta[:] = eta_new
    np.savez("./data/mech_run/adagrad", np.array(convergence_data), eta, eta)


def adadelta(eta, calculate_energy, calc_grad_theta, nx, ny):
    """
    http://arxiv.org/pdf/1212.5701v1.pdf
    """

    tolerance = 7.5e-9
    print_freq = 1

    time_var = time.time()

    # Calculate phi at the start, as it will remain constant
    phi = np.absolute(eta)

    eta_k = np.fft.fft2(eta)
    num_fts = 1

    # Starting values
    theta_last = np.angle(eta)
    g_last = calc_grad_theta(eta, eta_k)
    num_fts += 1

    e_g2 = np.zeros(theta_last.shape)
    e_dx2 = np.zeros(theta_last.shape)
    eps = 1e-2
    rho = 0.9

    convergence_data = []

    max_it = 200000
    for it in range(1, max_it+1):

        e_g2 = rho*e_g2+(1-rho)*g_last**2

        dtheta = -np.sqrt(e_dx2+eps)/np.sqrt(e_g2+eps)*g_last

        e_dx2 = rho*e_dx2+(1-rho)*dtheta**2

        theta_new = theta_last + dtheta

        eta_new = phi*np.exp(1j*theta_new)
        eta_k_new = np.fft.fft2(eta_new)
        num_fts += 1
        g_new = calc_grad_theta(eta_new, eta_k_new)
        num_fts += 1

        error = element_wise_norm(g_new, nx*ny)

        if it%print_freq == 0 or error < tolerance:
            energy = calculate_energy(eta_new, eta_k_new)
            dur = time.time()-time_var
            time_var = time.time()
            print("it: %d; en: %.16e; err: %.16e; fts: %d; time: %.1f" % (it, energy, error, num_fts, dur))
            convergence_data.append([it, energy, error, num_fts, dur])

        if error < tolerance:
            break

        theta_last = theta_new
        g_last = g_new

    eta[:] = eta_new
    #np.savez("./data/mech_run/adadelta", np.array(convergence_data), eta, eta)

