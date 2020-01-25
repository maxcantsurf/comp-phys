import numpy as np
import minimisation as mn
import matplotlib.pyplot as plt
import montecarlo as mc


def gp_2d(x, y):
    ''' Goldstein-Price function. No longer used in favour of the
    Rosenbrock function, but was used in for some initial validation.'''
    
    a = 1+(x+y+1)*(x+y+1)*(19-14*x+3*x*x-14*y+6*x*y+3*y*y)
    b = 30+(2*x-3*y)*(2*x-3*y)*(18-32*x+12*x*x+48*y-36*x*y+27*y*y)
    return(np.sqrt(a*b))

    
def gp_2d_x(x, y):
    return(gp_2d(x, y))
    
    
def gp_2d_y(y, x):
    return(gp_2d(x, y))

    
def rosenbrock_2d(x, y):
    ''' 2D Rosenbrock function, has a global minimum at (1, 1).'''
    
    f1 = 100*(y-x**2)**2 + (1-x)**2
    return(f1)
    

def rosenbrock_2d_x(x, y):
    ''' Helper function for the parabolic method.'''
    
    return(rosenbrock_2d(x, y))
    

def rosenbrock_2d_y(y, x):
    ''' Helper function for the parabolic method.'''
    
    return(rosenbrock_2d(x, y))


def rosenbrock_3d(x, y, z):
    ''' 3D Rosenbrock function, has a global minimum at (1, 1, 1).'''
    
    f1 = 100*(y-x**2)**2 + (1-x)**2
    f2 = 100*(z-y**2)**2 + (1-y)**2
    return(f1 + f2)
    

def test_2d(f, fx, fy, r0):
    ''' Tests the 3D minimnisers by appliying them to the 3D Rosenbrock 
    function whose minima is known to be (1, 1). It also plots the result
    and shows the paths taken by each of the minimisers.
    '''
    
    X, Y = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
    Z = np.zeros((len(X), len(Y)))
    i = 0
    while(i < len(X)):
        j = 0
        while(j < len(Y)):
            Z[i][j] = f(X[i][j], Y[i][j])
            j += 1
        i += 1
    
    fig = plt.figure(4)
    ax = fig.add_subplot(1, 1, 1)
    ax.contourf(X, Y, Z, 100, cmap = 'plasma', alpha = 0.2)
    ax.contour(X, Y, Z, 1000, cmap = 'plasma', alpha = 0.6)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.grid()
    
    
    lw = 3.0
    ls = '-'
    
    # Simulated annealing method
    
    path, r_min, f_min = mc.simulated_annealing(f, r0, (), notify = True)
    print(r_min[0], r_min[1])
    ax.plot(path[0], path[1], linewidth = lw, label = 'Monte Carlo', 
            color = 'red', linestyle = ls)
    
    # Univariate method
    
    path = mn.univariate_2d(fx, fy, r0, (), n = 10000, h = 1e-4)
    print(path[0][-1], path[1][-1])
    ax.plot(path[0], path[1], linewidth = lw, label = 'Univariate', 
            color = 'blue', linestyle = ls)
    
    # Gradient method
    
    path, ra = mn.gradient_method(f, r0, (), n = 100000, alpha = 1e-3, 
                                  h = 1e-8, accuracy = 1e-8)
    print(path[0][-1], path[1][-1])
    ax.plot(path[0], path[1], linewidth = lw, label = 'Gradient', 
            color = 'green', linestyle = ls)
    
    # Newton method
    
    path, ra = mn.newton_method(f, r0, (), n = 100000, alpha = 1e-1, 
                                  h = 1e-6, accuracy = 1e-8)
    print(path[0][-1], path[1][-1])
    ax.plot(path[0], path[1], linewidth = lw, label = 'Newton', 
            color = 'orange', linestyle = ls)
    
    # DFP quasi-Newton method
    
    path, ra = mn.quasi_newton(f, r0, (), n = 100000, alpha = 1e-4, 
                                  h = 1e-6, accuracy = 1e-8, 
                                  update_method = 'dfp')
    print(path[0][-1], path[1][-1])
    ax.plot(path[0], path[1], linewidth = lw, label = 'QN-DFP', 
            color = 'cornflowerblue', linestyle = ls)
    
    # BFGS quasi-Newton method
    
    path, ra = mn.quasi_newton(f, r0, (), n = 250000, alpha = 1e-3, 
                                  h = 1e-6, accuracy = 1e-8, 
                                  update_method = 'bfgs')
    print(path[0][-1], path[1][-1])
    ax.plot(path[0], path[1], linewidth = lw, label = 'QN-BFGS',
            color = 'limegreen', linestyle = ls)
    
    # Broyden quasi-Newton method
    
    path, ra = mn.quasi_newton(f, r0, (), n = 100000, alpha = 1e-4, 
                                  h = 1e-6, accuracy = 1e-8, 
                                  update_method = 'broyden')
    print(path[0][-1], path[1][-1])
    ax.plot(path[0], path[1], linewidth = lw, label = 'QN-Broyden',
            color = 'red', linestyle = ':')
    
    # SR1 quasi-Newton method
    
    path, ra = mn.quasi_newton(f, r0, (), n = 250000, alpha = 1e-3, 
                                  h = 1e-6, accuracy = 1e-8, 
                                  update_method = 'sr1')
    print(path[0][-1], path[1][-1])
    ax.plot(path[0], path[1], linewidth = lw, label = 'QN-SR1',
            color = 'black', linestyle = ':')
    
    ax.scatter(0, -1, color = 'black', s = 100)
    ax.legend()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    
def test_3d(function, r0):
    ''' Tests the 3D minimnisers by appliying them to the 3D Rosenbrock 
    function whose minima is known to be (1, 1, 1).
    '''
    
    # Gradient method
    
    path, ra = mn.gradient_method(function, r0, (), n = 100000, alpha = 1e-3, 
                                  h = 1e-6, accuracy = 1e-8)
    print(path[0][-1], path[1][-1])
    
    # Newton method
    
    path, ra = mn.newton_method(function, r0, (), n = 100000, alpha = 1e-1, 
                                  h = 1e-6, accuracy = 1e-8)
    print(path[0][-1], path[1][-1])
    
    # BFGS quasi-Newton method
    
    path, ra = mn.quasi_newton(function, r0, (), n = 250000, alpha = 1e-3, 
                                  h = 1e-6, accuracy = 1e-8, 
                                  update_method = 'dfp')
    print(path[0][-1], path[1][-1])
    
    # BFGS quasi-Newton method
    
    path, ra = mn.quasi_newton(function, r0, (), n = 250000, alpha = 1e-3, 
                                  h = 1e-6, accuracy = 1e-8, 
                                  update_method = 'bfgs')
    print(path[0][-1], path[1][-1])
    
    # BFGS quasi-Newton method
    
    path, ra = mn.quasi_newton(function, r0, (), n = 100000, alpha = 1e-3, 
                                  h = 1e-6, accuracy = 1e-8, 
                                  update_method = 'broyden')
    print(path[0][-1], path[1][-1])
    
    # BFGS quasi-Newton method
    
    path, ra = mn.quasi_newton(function, r0, (), n = 250000, alpha = 1e-3, 
                                  h = 1e-6, accuracy = 1e-8, 
                                  update_method = 'sr1')
    print(path[0][-1], path[1][-1])
