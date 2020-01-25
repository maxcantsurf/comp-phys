import numpy as np
import minimisation as mn
import montecarlo as mc
import result_validation as valid
import rootsolve as rs
import matplotlib.pyplot as plt
import base_functions as bf
import result_validation as rv
import os
import tkinter as tk

from tkinter import filedialog
from scipy.stats import kurtosis as kurt


def plot_prob(theta_1d, theta_2d, theta_3d, dmsq_1d, dmsq_2d, dmsq_3d, alpha, 
              lmda_meas, flux, energy, length):
    
    # We first calculate the predicted lambda profiles for the results from
    # the 1D, 2D and 3D minimisations
    
    lmda_pred_1d = bf.lmda(theta_1d, dmsq_1d, np.reciprocal(energy), flux,
                           energy, length)
    lmda_pred_2d = bf.lmda(theta_2d, dmsq_2d, np.reciprocal(energy), flux,
                           energy, length)
    lmda_pred_3d = bf.lmda(theta_3d, dmsq_3d, alpha, flux, energy, length)
    
    # The loop below generates a lambda profile from the resampled paramaters
    # when the generated data is averaged 100 times
    
    average = np.zeros(len(energy))
    N = 100
    n = 1
    while(n <= N):
        new_data = np.array(valid.generate_data(theta_3d, dmsq_3d, alpha, flux,
                                                energy, length))
        average = average*(n-1)/n + new_data/n
        n += 1
        
    # The rest of the code here is just boiler plate plotting code

    lw = 4.0
    fig = plt.figure(2)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(energy, lmda_meas, color = 'gray', linewidth = lw,
             label = r'Measured $\lambda$')
    ax.plot(energy, lmda_pred_1d, color = 'dodgerblue', linewidth = lw,
             label = r'Fitted $\lambda$ (1D)')
    ax.plot(energy, lmda_pred_2d, color = 'orange', linewidth = lw,
             label = r'Fitted $\lambda$ (2D)', linestyle = ':')
    ax.plot(energy, lmda_pred_3d, color = 'blue', linewidth = lw,
             label = r'Fitted $\lambda$ (3D)')
    ax.plot(energy, average, color = 'red', linewidth = lw, linestyle = ':',
             label = r'Resampled $\lambda$ (3D)')
    
    ax.set_xlabel('Energy (GeV)')
    ax.set_ylabel(r'$\lambda$')
    ax.grid()
    ax.legend()


def plot_nll_1d(lmda_meas, flux, energy, length):
    
    params = (2.4, np.reciprocal(energy), lmda_meas, flux, energy, length)
    thetas = np.linspace(np.pi/4, 0.9, 100)
    
    nlls = []
    for theta in thetas:
        nlls.append(bf.nll_theta(theta, *params))
    theta_min = mn.parabolic_1d(bf.nll_theta, [np.pi/4, 0.1 + np.pi/4], params,
                                error_estimate = True)
    nll_min = bf.nll_theta(theta_min, *params)
    
    theta_min_neg, theta_min_pos = bf.nll_theta_err(theta_min, params)
    nll_min_neg = bf.nll_theta(theta_min_neg, *params)
    nll_min_pos = bf.nll_theta(theta_min_pos, *params)
    theta_min_err = (theta_min_pos - theta_min_neg)/2
    
    lw = 3.0
    
    fig = plt.figure(3)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(thetas, np.array(nlls), color = 'blue', linewidth = lw)
    
    ax.axvline(np.pi/4, color = 'gray', linestyle = '--', linewidth = lw)
    ax.axvline(theta_min, color = 'r', linewidth = lw)
    ax.axvline(theta_min_neg, color = 'r', linestyle = '--', linewidth = lw)
    ax.axvline(theta_min_pos, color = 'r', linestyle = '--', linewidth = lw)
    
    ax.grid()
    ax.set_xlabel(r'$\theta_{23}$')
    ax.set_ylabel('NLL')
    
    print('')
    print(f'Left error bound  = {theta_min_neg}, (NLL = {nll_min_neg})')
    print(f'Right error bound = {theta_min_pos}, (NLL = {nll_min_pos})')
    print('')
    print('Difference in NLL between left/right error bounds = ', 
          abs(nll_min_pos - nll_min_neg))
    print('(Shold be close to zero)')
    print('')
    print('Difference between NLL min and NLL at right/left error bounds:')
    print(f'Right = {nll_min_pos - nll_min}')
    print(f'Left  = {nll_min_neg - nll_min}')
    print('(Should be close to 0.5)')
    print('')
    print(f'Theta minimum = {theta_min}' )
    print(f'Error (based on NLL change) = +/- {theta_min_err}')
    print(f'Error (based on last parabolic estimate) = +/-')

    
def plot_nll_2d(lmda_meas, flux, energy, length):
    
    # Params provide the input data etc to fit. In the 2D case we do not 
    # include the effect of cross section energy dependence. Since the
    # dependence is just a factor of alpha*energy, we just set alpha to
    # be the reciprocal of energy to give a net effect of multiplying by
    # unity, i.e. no cross section energy dependence
    
    params = (np.reciprocal(energy), lmda_meas, flux, energy, length)
    
    # Now we set up an x-y mesh grid on which to plot the 2D function and
    # the minimisation paths
    # X and Y store the input values of theta and dmsq. We iterate through the
    # (x, y) pairs and set the corresponding element in Z to the value of
    # the 2D NLL at that point 
    
    x_bound = (0.7, 1.0)
    y_bound = (2.0, 3.0)
    
    X, Y = np.meshgrid(np.linspace(*x_bound, 100), np.linspace(*y_bound, 100))
    Z = np.zeros((len(X), len(Y)))
    
    i = 0
    while(i < len(X)):
        j = 0
        while(j < len(Y)):
            Z[i][j] = bf.nll(X[i][j], Y[i][j], *params)
            j += 1
        i += 1
        
    # We set the line width lw and starting position r0
    # The next few lines are just boiler plate. Note we do two plots,
    # a countour line plot and countourf plot to fill in between the lines
    
    lw = 3.0
    r0 = [0.9, 2.7]
    
    fig = plt.figure(4)
    ax = fig.add_subplot(1, 1, 1)
    ax.contourf(X, Y, Z, 100, cmap = 'plasma', alpha = 0.5)
    ax.contour(X, Y, Z, 100, cmap = 'plasma')
    ax.set_xlabel(r'$\theta_{23}$')
    ax.set_ylabel(r'$\Delta m^2_{23} \mathrm{eV}^2 \times 10^{-3}$')
    
    # The simulated annealing method
    
    path, r_min, f_min = mc.simulated_annealing(bf.nll, r0, params)
    theta_path, dmsq_path  = path[0], path[1]
    theta_min, dmsq_min = theta_path[-1], dmsq_path[-1]
    ax.plot(theta_path, dmsq_path, color = 'darkorange', label = 'Monte Carlo',
            linewidth = lw)
    
    # The univariate method
    
    print('')
    path = mn.univariate_2d(bf.nll_theta, bf.nll_dmsq, r0, params, h = 1e-4)
    theta_path, dmsq_path  = path[0], path[1]
    theta_min, dmsq_min = theta_path[-1], dmsq_path[-1]
    ax.plot(theta_path, dmsq_path, color = 'dodgerblue', label = '2D Parabolic',
            linewidth = lw)
    
    print(f'Univariate method results: theta = {theta_min}, dmsq = {dmsq_min}')
    
    # The gradient method
    
    print('')
    path, ra = mn.gradient_method(bf.nll, r0, params, n = 1000, alpha = 1e-4, 
                                  h = 1e-10, accuracy = 1e-8, notify = True)
    theta_path, dmsq_path  = path[0], path[1]
    theta_min, dmsq_min = theta_path[-1], dmsq_path[-1]
    ax.plot(theta_path, dmsq_path, color = 'b', label = 'Gradient Method',
            linewidth = lw)
    print('')
    print(f'Gradient method results: theta = {theta_min}, dmsq = {dmsq_min}')
    
    # The Newton method
    
    print('')
    path, ra = mn.newton_method(bf.nll, r0, params, n = 100, h = 1e-6, 
                                accuracy = 1e-8, alpha = 0.1, notify = True)
    theta_path, dmsq_path  = path[0], path[1]
    theta_min, dmsq_min = theta_path[-1], dmsq_path[-1]
    ax.plot(theta_path, dmsq_path, color = 'r', label = 'Newton Method',
            linewidth = lw)
    print('')
    print(f'Newton method results: theta = {theta_min}, '
          + f'dmsq = {dmsq_min}')
    print('')
    
    # The quasi-Newton DFP method
    
    print('')
    path, ra = mn.quasi_newton(bf.nll, r0, params, n = 100, h = 1e-6, 
                                accuracy = 1e-8, alpha = 1e-4, notify = True,
                                update_method = 'dfp')
    theta_path, dmsq_path  = path[0], path[1]
    theta_min, dmsq_min = theta_path[-1], dmsq_path[-1]
    ax.plot(theta_path, dmsq_path, color = 'r', label = 'DFP Method',
            linewidth = lw, linestyle = ':')
    print('')
    print(f'Newton method results: theta = {theta_min}, '
          + f'dmsq = {dmsq_min}')
    print('')
    
    # The four points here are points at which the NLL function is equal
    # to the NLL minimum + 0.5, along the x and y directions. They essentailly
    # define the points at which top draw the error bars between
    
    nx, px, ny, py = bf.nll_err_2d(theta_min, dmsq_min, params)
    
    # From these values we can work out the error in the quantities, 
    # we have assumed that the error in both directions is the same 
    
    theta_err = (px-nx)/2
    dmsq_err  = (py-ny)/2
    
    # Now we can go ahead and plot the error bars
    
    ax.plot([theta_min - theta_err, theta_min + theta_err], 
            [dmsq_min, dmsq_min], color = 'black',
            linewidth = lw)
    ax.plot([theta_min, theta_min], 
            [dmsq_min - dmsq_err, dmsq_min + dmsq_err], color = 'black',
            linewidth = lw)
    
    # Now we just go ahead and print out the results
    
    print(f'theta error = +/- {theta_err}')
    print(f'dmsq error  = +/- {dmsq_err}')
    ax.grid()
    ax.legend()
    ax.set_xlim([*x_bound])
    ax.set_ylim([*y_bound])


def plot_nll_err_3d(x_min, y_min, z_min, params):
    ''' Finds the error at the given minimum and plots this onto the given 
    axes. Also returns the error in each parameter.
    '''
    
    # Start by setting up the parameters for each of the helped functions
    
    nll_min = bf.nll(x_min, y_min, z_min, *params)
    x_params = y_min, z_min, *params, nll_min
    y_params = x_min, z_min, *params, nll_min
    z_params = x_min, y_min, *params, nll_min
    
    # To find the error for each parameter we just do a root solve for
    # an nll helper function which foxes the other two variables and is
    # equal to the nll minus the nll miniumum minus 0.5.
    # We find the error in the +ve/-ve direction from the minimum by starting
    # the root solving algorithm either above/below 
    
    xp = rs.root_find_secant(bf.nll_theta_shifted, x_params, x_min + 0.01)
    xn = rs.root_find_secant(bf.nll_theta_shifted, x_params, x_min - 0.01)
    yp = rs.root_find_secant(bf.nll_dmsq_shifted, y_params, y_min + 0.01)
    yn = rs.root_find_secant(bf.nll_dmsq_shifted, y_params, y_min - 0.01)
    zp = rs.root_find_secant(bf.nll_alpha_shifted, z_params, z_min + 0.01)
    zn = rs.root_find_secant(bf.nll_alpha_shifted, z_params,  z_min - 0.01)

    # Too find the error we take the difference between the two points where
    # the nll crossed its minimum + 0.5 and divide by two to get an average
    # We do not take the absolute value as we should expect the root found
    # on the positive side to be bigger, so we can look out for negative
    # negative errors to check for bugs/if something went wrong
    
    x_err = (xp - xn)/2
    y_err = (yp - yn)/2
    z_err = (zp - zn)/2
    
    return(x_err, y_err, z_err)


def plot_nll_3d(x_min, y_min, z_min, xb, yb, zb, params, r0):
    ''' Carries out a 3D minimisation using the gradient, Newton and
    quasi-Newton (DFP) methods. It then plots three 2D slices of the 3D space
    about the provided minimum point and plots the paths taken by the various
    methods onto these, along with the associated uncertainty in the minimum 
    point.
    Note that this method is more to demonstrate the 3D minimisers as we
    require the minimium to be known to good precision before hand so we can 
    plot out the space around it. Here the minimisers are set to a lower 
    accuracy so it can be executed quickly.
    The plot is conducted in a nice way such that the axes of each plot
    line up.
    
    Keyword arguments:
        x_min, y_min, z_min -- The values of theta, dmsq, alpha respectively
        which minimise the value of the NLL, these are should be precalculated
        to a high precision
        xb, yb, zb -- The bounds over which to plot the 3D function for each
        of the respective variables.
        params -- The additional paramaters for the NLL function
        r0 -- The starting point for the minimisers
        
    '''
    
    N = 100
    cmap = 'plasma'
    linebr = '==================================================='
    
    # Here we are just setting up the plotting environment for the 3D case

    xyx, xyy = np.meshgrid(np.linspace(*xb, N), np.linspace(*yb, N))
    xzx, xzz = np.meshgrid(np.linspace(*xb, N), np.linspace(*zb, N))
    yzy, yzz = np.meshgrid(np.linspace(*yb, N), np.linspace(*zb, N))
    
    xyf = np.zeros((len(xyx), len(xyy)))
    xzf = np.zeros((len(xzx), len(xzz)))
    yzf = np.zeros((len(yzy), len(yzz)))
    
    i = 0
    while(i < len(xyx)):
        j = 0
        while(j < len(xyy)):
            xyf[i][j] = bf.nll(xyx[i][j], xyy[i][j], z_min, *params)
            j += 1
        i += 1
        
    i = 0
    while(i < len(xzx)):
        j = 0
        while(j < len(xzz)):
            xzf[i][j] = bf.nll(xzx[i][j], y_min, xzz[i][j], *params)
            j += 1
        i += 1
    
    i = 0
    while(i < len(yzy)):
        j = 0
        while(j < len(yzz)):
            yzf[i][j] = bf.nll(x_min, yzy[i][j], yzz[i][j], *params)
            j += 1
        i += 1
    
    fig = plt.figure(5)
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    
    ax1.contourf(xyx, xyy, xyf, N, cmap = cmap, alpha = 0.5)
    ax2.contourf(yzz, yzy, xzf, N, cmap = cmap, alpha = 0.5)
    ax3.contourf(xzx, xzz, yzf, N, cmap = cmap, alpha = 0.5)
    
    ax1.contour(xyx, xyy, xyf, N, cmap = cmap)
    ax2.contour(yzz, yzy, xzf, N, cmap = cmap)
    ax3.contour(xzx, xzz, yzf, N, cmap = cmap)
    
    #ax1.set_xlabel(r'$\theta_{23}$')
    ax1.set_ylabel(r'$\Delta m^2_{23} \mathrm{eV}^2 \times 10^{-3}$')
    ax2.set_xlabel(r'$\alpha$')
    #ax2.set_ylabel(r'$\Delta m^2_{23}$')
    ax3.set_xlabel(r'$\theta_{23}$')
    ax3.set_ylabel(r'$\alpha$')
    
    ax1.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    
    ax1.scatter(x_min, y_min, color = 'black')
    ax2.scatter(z_min, y_min, color = 'black')
    ax3.scatter(x_min, z_min, color = 'black')
    
    ax1.set_xlim([*xb])
    ax1.set_ylim([*yb])
    ax2.set_xlim([*zb])
    ax2.set_ylim([*yb])
    ax3.set_xlim([*xb])
    ax3.set_ylim([*zb])
    
    x_err, y_err, z_err = plot_nll_err_3d(x_min, y_min, z_min, params)
    
    c = 'black'
    lw = 3.0
    
    # Now we pot and print the calculated errors as error bars/crosshairs
    # Note the error is calculated using the given minium and the minimisations
    # are only carries out here to demonstarte they give the same minimum

    ax1.errorbar(x_min, y_min, xerr = x_err, yerr = y_err,
                 elinewidth = lw, ecolor = c)
    ax2.errorbar(z_min, y_min, xerr = z_err, yerr = y_err,
                 elinewidth = lw, color = c)
    ax3.errorbar(x_min, z_min, xerr = x_err, yerr = z_err,
                 elinewidth = lw, color = c)
        
    print('Errors around given minimum:')
    print(f'theta = {x_min} +/- {x_err}')
    print(f'dmsq  = {y_min} +/- {y_err}')
    print(f'alpha = {z_min} +/- {z_err}')
    
    # Gradient descent method
    # First we carry out the minimisation
    
    print(linebr)
    
    path, ra = mn.gradient_method(bf.nll, r0, params, alpha = 1e-4, n = 1000,
                                  h = 1e-8, accuracy = 1e-8)
    
    # Now extract the path of each coordinate and the minimum values
    
    x_path, y_path, z_path = path[0], path[1], path[2]
    x_min, y_min, z_min = x_path[-1], y_path[-1], z_path[-1]
    nll_min = bf.nll(x_min, y_min, z_min, *params)
    
    # Now just plot the path and print the parameters 
    
    lw = 4.0
    ls = ':'
    
    ax1.plot(x_path, y_path, color = 'r', linewidth = lw, label = 'Gradient')
    ax2.plot(z_path, y_path, color = 'r', linewidth = lw, label = 'Gradient')
    ax3.plot(x_path, z_path, color = 'r', linewidth = lw, label = 'Gradient')
    
    print('')
    print(f'Gradient descent method results:')
    print(f'theta = {x_min}, dmsq = {y_min}, alpha = {z_min}, nll = {nll_min}')
    print('')
    
    # As a final step we calculate the Hessian at the calculated minimum and
    # determine if its definiteness. If it is positive definite it is a 
    # minimum, otherwise it is either a saddle point or a maximum
    
    rv.check_definiteness_3d(bf.nll, np.array([x_min, y_min, z_min]), params)
    
    print(linebr)
    
    # Simmulated annealing method
    # Process is the same as above 
    
    path, r_min, f_min = mc.simulated_annealing(bf.nll, r0, params)
    
    x_path, y_path, z_path = path[0], path[1], path[2]
    x_min, y_min, z_min = x_path[-1], y_path[-1], z_path[-1]
    nll_min = bf.nll(x_min, y_min, z_min, *params)
    
    lw = 4.0
    ls = ':'
    
    ax1.plot(x_path, y_path, color = 'salmon', linewidth = lw,
             label = 'Monte Carlo')
    ax2.plot(z_path, y_path, color = 'salmon', linewidth = lw,
             label = 'Monte Carlo')
    ax3.plot(x_path, z_path, color = 'salmon', linewidth = lw, 
             label = 'Monte Carlo')
    
    print('')
    print(f'Simmulated annealing results:')
    print(f'theta = {x_min}, dmsq = {y_min}, alpha = {z_min}, nll = {nll_min}')
    print('')
    
    rv.check_definiteness_3d(bf.nll, np.array([x_min, y_min, z_min]), params)
    
    print(linebr)
    
    # Newton method
    # Process is the same as above 
    
    path, ra = mn.newton_method(bf.nll, r0, params, alpha = 1e-3, n = 2500,
                                  h = 1e-6, accuracy = 1e-8,)
    
    x_path, y_path, z_path = path[0], path[1], path[2]
    x_min, y_min, z_min = x_path[-1], y_path[-1], z_path[-1]
    nll_min = bf.nll(x_min, y_min, z_min, *params)
    
    ax1.plot(x_path, y_path, color = 'b', linewidth = lw, label = 'Newton')
    ax2.plot(z_path, y_path, color = 'b', linewidth = lw, label = 'Newton')
    ax3.plot(x_path, z_path, color = 'b', linewidth = lw, label = 'Newton')
    
    print('')
    print(f'Newton method results:')
    print(f'theta = {x_min}, dmsq = {y_min}, alpha = {z_min}, nll = {nll_min}')
    print('')
    
    rv.check_definiteness_3d(bf.nll, np.array([x_min, y_min, z_min]), params)
    
    print(linebr)
    
    # Quasi-Newton method
    # Process is the same as above
    
    path, ra = mn.quasi_newton(bf.nll, r0, params, alpha = 1e-4, n = 1000,
                                  h = 1e-8, accuracy = 1e-8, 
                                  update_method = 'dfp')
    
    x_path, y_path, z_path = path[0], path[1], path[2]
    x_min, y_min, z_min = x_path[-1], y_path[-1], z_path[-1]
    nll_min = bf.nll(x_min, y_min, z_min, *params)
    
    ax1.plot(x_path, y_path, color = 'b', linewidth = lw, linestyle = ls,
             label = 'QN DFP')
    ax2.plot(z_path, y_path, color = 'b', linewidth = lw, linestyle = ls,
             label = 'QN DFP')
    ax3.plot(x_path, z_path, color = 'b', linewidth = lw, linestyle = ls,
             label = 'QN DFP')
    
    print('')
    print(f'Quasi-Newton (DFP) method results:')
    print(f'theta = {x_min}, dmsq = {y_min}, alpha = {z_min}, nll = {nll_min}')
    print('')
    
    rv.check_definiteness_3d(bf.nll, np.array([x_min, y_min, z_min]), params)
    
    print(linebr)
    
    ax2.legend()
    
    
def plot_resample(x_min, y_min, z_min, lmda, flux, energy, length, avg = 100):
    ''' Takes a set of chosen data files producded from the resampling methods
    and pulls them together to calculate the mean values of the resampled 
    parameters. It also produces a plot for each parameter and the NLL. These
    values can then be compared against the values used to generate the data
    which is given as input along with the normal additional NLL parameters.
    '''
    
    x_list = []
    y_list = []
    z_list = []
    nll_list = []
    
    # We first open up a file prompt which allows the user to select which 
    # files they would like to include into their data set
    
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(initialdir = os.getcwd()
        + f'/resample_data/avg{str(avg)}/')
    
    # now we just loop through all of the files that were selected and pull 
    # all of the data into a single set of lists for each parameter and the 
    # NLL
    
    for file in files:
        new_x, new_y, new_z, new_nll = np.loadtxt(file)
        x_list += list(new_x)
        y_list += list(new_y)
        z_list += list(new_z)
        nll_list += list(new_nll)
        
    # Now we calculate some important paramaters for the resampled paramaters 
    # so they can be compared to the source/generating parameters
        
    x_mean, x_std, x_kurt = np.mean(x_list), np.std(x_list), kurt(x_list)
    y_mean, y_std, y_kurt = np.mean(y_list), np.std(y_list), kurt(y_list)
    z_mean, z_std, z_kurt = np.mean(z_list), np.std(z_list), kurt(z_list)
    nll_mean, nll_std = np.mean(nll_list), np.std(nll_list)
    true_nll = bf.nll(x_min, y_min, z_min, lmda, flux, energy, length)
    
    # The rest is just boiler plate code in which we print out the results
    # before plotting them graphically as histograms
    
    print(f'Total dataset size n = {len(x_list)}')
    print('')
    print(f'True theta = {x_min}')
    print(f'Mean theta = {x_mean} +/- {x_std}')
    print(f'Mean theta bounds = {x_mean - x_std} -> {x_mean + x_std}')
    print(f'theta kurtosis = {x_kurt}')
    print('')
    print(f'True dmsq = {y_min}')
    print(f'Mean dmsq = {y_mean} +/- {y_std}')
    print(f'Mean dmsq bounds = {y_mean - y_std} -> {y_mean + y_std}')
    print(f'dmsq kurtosis = {y_kurt}')
    print('')
    print(f'True alpha = {z_min}')
    print(f'Mean alpha = {z_mean} +/- {z_std}')
    print(f'Mean alpha bounds = {z_mean - z_std} -> {z_mean + z_std}')
    print(f'alpha kurtosis = {z_kurt}')
    print('')
    print(f'True nll = {true_nll}')
    print(f'Mean nll = {nll_mean} +/- {nll_std}')
    print(f'Mean nll bounds = {nll_mean - nll_std} -> {nll_mean + nll_std}')
    
    
    fig = plt.figure(81)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    #ax4 = fig.add_subplot(2, 2, 4)
    
    bins = 100
    lw = 3.0
    
    ax1.hist(x_list, bins = bins, density = True)
    ax2.hist(y_list, bins = bins, density = True)
    ax3.hist(z_list, bins = bins, density = True)
    #ax4.hist(nll_list, bins = bins)
    
    ax1.axvline(x_min, color = 'b', linewidth = lw)
    ax2.axvline(y_min, color = 'b', linewidth = lw)
    ax3.axvline(z_min, color = 'b', linewidth = lw)
    #ax4.axvline(true_nll, color = 'b', linewidth = lw)
    
    ax1.axvline(x_mean, color = 'r', linewidth = lw, linestyle = ':')
    ax2.axvline(y_mean, color = 'r', linewidth = lw, linestyle = ':')
    ax3.axvline(z_mean, color = 'r', linewidth = lw, linestyle = ':')
    #ax4.axvline(nll_mean, color = 'r', linewidth = lw)
    
    ax1.axvline(x_mean + x_std, color = 'r', linewidth = lw, linestyle = '--')
    ax2.axvline(y_mean + y_std, color = 'r', linewidth = lw, linestyle = '--')
    ax3.axvline(z_mean + z_std, color = 'r', linewidth = lw, linestyle = '--')
    #ax4.axvline(nll_mean + nll_std, color = 'r', linewidth = lw, linestyle = '--')
    
    ax1.axvline(x_mean - x_std, color = 'r', linewidth = lw, linestyle = '--')
    ax2.axvline(y_mean - y_std, color = 'r', linewidth = lw, linestyle = '--')
    ax3.axvline(z_mean - z_std, color = 'r', linewidth = lw, linestyle = '--')
    #ax4.axvline(nll_mean - nll_std, color = 'r', linewidth = lw, linestyle = '--')
    
    ax1.set_xlabel(r'$\theta_{23}$')
    ax2.set_xlabel(r'$\Delta m^2_{23} \mathrm{eV}^2 \times 10^{-3}$')
    ax3.set_xlabel(r'$\alpha$')
    #ax4.set_xlabel('Minimum NLL')
    
    ax1.set_ylabel('Density')
    
    ax2.get_yaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    

def plot_resample_var_avg(x_min, y_min, z_min):
    ''' Takes a set of chosen data files producded from the resampling methods
    and pulls them together to calculate the mean values of the resampled 
    parameters. It also produces a plot for each parameter and the NLL. These
    values can then be compared against the values used to generate the data
    which is given as input along with the normal additional NLL parameters.
    '''
    
    # First define all of the lists which will hold the mean resampled value
    # for each of the different number of averaging steps, along with the list
    # avg_list which holds the corresponding number of averaging steps which
    # were done
    
    avg_list = []
    x_means, x_stds = [], []
    y_means, y_stds = [], []
    z_means, z_stds = [], []
    nll_means, nll_stds = [], []
    
    # Get the current working directory so we can work relative to it
    
    directory = os.fsencode(os.getcwd() + '/resample_data/')
    
    # All of the folders in the resample_data folder hold resampling data for 
    # different numbers of averaging steps. For example avg10 hols all of the
    # resampled values which were fitted to a data set which was the avarge of
    # ten generated data sets
    
    for folder in os.listdir(directory):
        foldername = os.fsdecode(folder)
        avg = int(foldername.replace('avg', ''))
        x_list = []
        y_list = []
        z_list = []
        nll_list = []
        directory_new = os.fsencode(os.getcwd() + f'/resample_data/avg{avg}/')
        
        # Now we loop over all of the files within the folder and pull all of 
        # the lists of resampled values into one big list which we then
        # can then take the mean of and compare to the source value etc
        
        for file in os.listdir(directory_new):
            filename = os.fsdecode(file)
            filename = os.getcwd() + f'/resample_data/avg{avg}/{filename}'
            new_x, new_y, new_z, new_nll = np.loadtxt(filename)
            x_list += list(new_x)
            y_list += list(new_y)
            z_list += list(new_z)
            nll_list += list(new_nll)
            
        # Now we have pulled all of the lists in the folder into a single list
        # we take the mean to find the mean resampled value for when this
        # number of averaging steps is used
        # We also record the standard deviation for good measure
        
        avg_list.append(avg)
        x_means.append(np.mean(x_list))
        y_means.append(np.mean(y_list))
        z_means.append(np.mean(z_list))
        nll_means.append(np.mean(nll_list))
        
        x_stds.append(np.std(x_list))
        y_stds.append(np.std(y_list))
        z_stds.append(np.std(z_list))
        nll_stds.append(np.std(nll_list))
        
        # Here we print the percentage difference between the mean resampled
        # value and the source value which is given as input. Since we are
        # only really concerned with alpha we neglect comparing the other
        # two resampled means to their source values
        
        percentage_error = (np.mean(z_list)/z_min - 1)*100
        print(f'avg = {avg}, % error = {percentage_error}')
        
    # Once we have calculated the mean resampled values for eahc of the
    # different number of averaging steps we plot the results so we can get an
    # idea of how quickly the mean resampled value converges to the source 
    # value as we increase the number of averaging steps
        
    fig = plt.figure(20)
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(avg_list, z_means, yerr = z_stds, linewidth = 4.0, color = 'b',
                ecolor = 'gray', fmt = 'none')
    ax.scatter(avg_list, z_means, color = 'b', s = 600, marker = 'x')
    ax.axhline(z_min, color = 'red', linestyle = ':', linewidth = 4.0)
    ax.set_xlabel('Number of Averages')
    ax.set_ylabel(r'$\alpha$')
    ax.grid()
    
    
def plot_global_search_2d(xb, yb, nx, ny, lmda_meas, flux, energy, length):
    
    params = (np.reciprocal(energy), lmda_meas, flux, energy, length)
    
    x_range = np.linspace(*xb, nx)
    y_range = np.linspace(*yb, ny)
    
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros((len(X), len(Y)))
    
    i = 0
    while(i < len(X)):
        j = 0
        while(j < len(Y)):
            Z[i][j] = bf.nll(X[i][j], Y[i][j], *params)
            j += 1
        i += 1
    
    fig = plt.figure(4)
    ax = fig.add_subplot(1, 1, 1)
    ax.contourf(X, Y, Z, 500, cmap = 'plasma', alpha = 0.5)
    ax.contour(X, Y, Z, 500, cmap = 'plasma')
    ax.set_xlabel(r'$\theta_{23}$')
    ax.set_ylabel(r'$\Delta m^2_{23} \mathrm{eV}^2 \times 10^{-3}$')
    
    paths, path_min_global, r_min_global = mc.global_search_2d(bf.nll, params,
                                                               xb, yb, nx, ny)
    
    for path in paths:
        ax.plot(path[0], path[1], color = 'black')
    
