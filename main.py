import numpy as np
import result_validation as rv
import minimiser_validation as mv
import plotting as plot
import matplotlib.pyplot as plt
import warnings
import base_functions as bf
import montecarlo as mc
import minimisation as mn


if(__name__ == '__main__'):
    
    # Boiler plate code
    
    rc = {'font.size': 24, 
          'mathtext.fontset': 'cm',
          'legend.fontsize': 16}
    plt.rcParams.update(rc)
    warnings.filterwarnings('ignore', category = DeprecationWarning)
    
    # Now we define some constants and import the experimental data
    # We then package these together as params to avoid having to 
    # constantly re-write them
    
    length = 295.0
    lmda_meas = np.loadtxt('bins.txt')
    flux = np.loadtxt('flux.txt')
    energy = np.linspace(0.025, 9.975, 200)
    params = (lmda_meas, flux, energy, length)
    
    # These are parameter estimates which have been pre-calculated using
    # high accuracy runs of the gradient method
    
    theta_1d = 0.8147342429350978
    theta_2d = 0.8147154425019835
    theta_3d = 0.8180010693995314  
    dmsq_1d = 2.4
    dmsq_2d = 2.4162094018676727
    dmsq_3d = 2.525603221456363
    alpha_3d = 1.415869315096163
    
    xb = (0.6, 1.0)
    yb = (1.6, 3.0)
    zb = (1.0, 1.8)
    
    # Arguments for minimiser params are:
    # n, alpha, h, accuracy, notify
    
    minimiser_params = (1000, 1e-4, 1e-8, 1e-8, False)


def plot_data():
    ''' Produces a plot of the raw data along with the theoretical profiles
    produced when using the fitted parameters.
    '''
    
    plot.plot_prob(theta_1d, theta_2d, theta_3d, dmsq_1d, dmsq_2d, dmsq_3d,
                   alpha_3d, *params)
    
    
def minimisation_1d():
    ''' Carries out single parameter estimation for the case where only the 
    mixing angle is varied while dmsq is fixed at 2.4. Cross-section is
    independent of energy.
    '''
    
    plot.plot_nll_1d(*params)
    

def minimisation_2d():
    ''' Carries out double-parameter estimation for the case where we wish to
    fit both theta and dmsq. The cross section is again taken to be independent
    of energy.
    '''
    
    plot.plot_nll_2d(*params)
    
    
def minimisation_3d():
    ''' Carries out triple-parameter extimation when we are also including the
    linear scaling of the cross section with energy. 
    '''
    
    r0 = np.array([0.95, 2.2, 1.1])
    plot.plot_nll_3d(theta_3d, dmsq_3d, alpha_3d, xb, yb, zb, params, r0)
    
    
def minimisation_4d():
    ''' Performs the same task as the 3D fitting case except we also fit an 
    additional 'offset' term, beta. The puropose of this routine is to 
    demonstate such an additional parameter is unnessesary. 
    '''
    
    r0 = np.array([0.85, 2.4, 1.4, 0.1])
    path, ra = mn.gradient_method(bf.nll_4d, r0, params)
    theta = path[0][-1]
    dmsq  = path[1][-1]
    alpha = path[2][-1]
    beta  = path[3][-1] 
    nll = bf.nll_4d(theta, dmsq, alpha, beta, *params)
    print('')
    print('4D Minimisation complete: ')
    print('')
    print(f'theta = {theta}')
    print(f'dmsq = {dmsq}')
    print(f'alpha = {alpha}')
    print(f'beta = {beta}')
    print(f'nll = {nll}')
    
    
def global_min_2d():
    ''' Plots the paths taken by a 2D global search. '''
    
    plot.plot_global_search_2d((0.5, 1.0), (2.0, 3.0), 4, 4, *params)
    
    
def global_min_3d():
    ''' Carries out a global search in 3D space using the direct gradient
    descent method.'''
    
    path, ra = mn.global_gradient_3d(bf.nll, (0.75, 1.0), (2.0, 3.0), 
                                     (1.0, 2.0), 5, 5, 5, params, ())
    
def mc_global_min_3d():
    ''' Does the same as the global_min_3d() method except the Monte Carlo
    simulated annealing method is used instead.
    '''
    
    r_min, f_min = mc.global_search_3d(bf.nll, params, (0.75, 1.0), (2.0, 3.0), 
                                     (1.0, 2.0), 5, 5, 5)
    
    
def test_minimisers_2d():
    ''' Tests all of the 2D minimisers on the 2D Rosenbrock function and
    plots the result, and prints the performance data. 
    '''
    
    mv.test_2d(mv.rosenbrock_2d, mv.rosenbrock_2d_x, mv.rosenbrock_2d_y,
               np.array([-1.5, -1.5]))

 
def test_minimisers_3d():
    ''' Tests all of the 3D minimisers on the 3D Rosenbrock function and
    plots the result, and prints the performance data. 
    '''
    
    mv.test_3d(mv.rosenbrock_3d, np.array([0.0, 0.0, 0.0]))  


def resample_singlethread():
    ''' Generates resampled data similarly to resample_multithread, except
    only uses a single thread, and so is much slower. '''
    
    rv.resample(theta_3d, dmsq_3d, alpha_3d, flux, energy, length, 
                minimiser_params, num_runs = 100)
    
    
def resample_multithread():
    ''' Generates a new data set using the source parameters specified in the
    preamble. It is multithreaded to allow the process to be done much quicker.
    
    See the docstring for the method itself for more info.
    If you are on windows, make sure when running this method you execute in an
    external system console otherwise you will not see the print output
    from the other threads.
    '''
    
    print('If on windows make sure you are using an external system terminal!')
    rv.resample_multithread(theta_3d, dmsq_3d, alpha_3d, flux, energy, length,
                            minimiser_params, num_threads = 6, m = 1000,
                            num_averages = 9)
    input('Press ENTER to exit...')
    
    
def plot_resample():
    ''' Opens a dialog which allows you to choose which files you would like
    to see resampling data from. This data is then plotted and compared against
    the source parameters which we provide.
    '''
    
    plot.plot_resample(theta_3d, dmsq_3d, alpha_3d, *params)
    
    
def plot_resample_var_avg():
    ''' Generates a plot which plots how the mean resampled value varies 
    with respect to the source value as we vary the number of averaging steps.
    '''
    
    plot.plot_resample_var_avg(theta_3d, dmsq_3d, alpha_3d)
  

if(__name__ == '__main__'):
    print('Welcome! Here is a list of available commands:')
    print('')
    print('plot_data() -- Plots the raw data along with various fits to it.')
    print('')
    print('minimisation_1d() -- Carry out a 1D minimisation of the NLL.')
    print('minimisation_2d() -- Carry out a 2D minimisation of the NLL.')
    print('minimisation_3d() -- Carry out a 3D minimisation of the NLL.')
    print('minimisation_4d() -- Carry out a 4D minimisation of the NLL.')
    print('')
    print('test_minimisers_2d() -- Test the minimisers on a 2D test function.')
    print('test_minimisers_3d() -- Test the minimisers on a 2D test function.')
    print('')
    print('plot_resample() -- Plots the resampled means from a chosen file.')
    print('resample_singlethread() -- Carries out single-threaded resampling.')
    print('resample_multithread() -- Carries out multi-threaded resampling.')
    print('plot_resample_var_avg() -- Shows how resampling varies as the '
          + 'number of averaging steps is varied.')
    
    # Commented here for easy copy-pasting into the terminal...
    
    #plot_data()
    
    #minimisation_1d()
    #minimisation_2d()
    #minimisation_3d()
    #minimisation_4d()
    
    #global_min_2d()
    #global_min_3d()
    
    #global_min_2d()
    #global_min_3d()
    #mc_global_min_3d()
    
    #test_minimisers_2d()
    #test_minimisers_3d()
    
    #plot_resample()
    #resample_singlethread()
    #resample_multithread()
    #plot_resample_var_avg()
    
    #global_min_2d()
    
    # When using resample_multithread, if you are running Windows make sure to
    # set the console to be an external system console otherwise you will not
    # be able to see the progress output
    # To do this in Spyder go to Run -> Configuration per file -> Execute in
    # an external system terminal
