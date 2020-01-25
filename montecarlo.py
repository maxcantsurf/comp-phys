import numpy as np
import time


def simulated_annealing(f, r0, params, generator = 'gaussian',
                        sigma = 1e-2, T_start = 1e-3, T_end = 1e-5,
                        n = 10000, notify = False):
    ''' Implementation of a Monte-Carlo based simmulated annealing routine 
    which uses a Metropolic-style probabalistic walk to move torwards the 
    minimimum. 
    
    Keyword arguments:
        generator -- The PDF generator to be used to generate new steps, 
        possible choices are: 'gaussian'
        sigma -- The standard deviation to use for the gaussian gnerator
        T_start, T_end -- The start and end 'temperatures' to use in the
        exponential factor of the accept/reject probability. The algorithm
        will linearly move from T_start --> T_end as it progresses
        n -- The maximum number of steps to make
    '''
    
    r = r0
    T = T_start
    delta_T = (T_start - T_end)/n
    path = []
    i = 0
    f_min = np.inf
    r_min = r0
    t0 = time.time()
    
    while(i < n):
        
        # Here we generate the proposed step
        
        g = None
        if(generator == 'gaussian'):
            g = np.random.normal(r, sigma) - r
            
        f_r = f(*r, *params)
        
        # We check if the current position is the lowest minimum thus far
        
        if(f_r < f_min):
            f_min = f_r
            r_min = r
            
        # delta_E is the change in the function between the current and
        # proposed next step
        
        delta_E = f(*(r+g), *params) - f_r
        p_acc = 0
        
        # If the funciton chnage was negative, we accept the step with
        # probability one. If it increased the function, we still have some
        # finite probability to accept the step
        
        if(delta_E <= 0):
            p_acc = 1
        else:
            p_acc = np.exp(-delta_E/T)
            
        if(np.ma.is_masked(p_acc)):
            p_acc = 0
            
        # A single probabilistic decision is equivalent to a binomial with
        # just one trial
        
        if(np.random.binomial(1, p_acc)):
            path.append(r)
            r = r + g
        
        T += delta_T
        i += 1
    
    t1 = time.time()
        
    if(notify):
            print(f'Simulated annealing done. Completed {i} iterations.')
            print(f'Time taken: {1000*(t1-t0)}ms ({1000*(t1-t0)/i}s per loop)')
            
    return(np.transpose(np.array(path)), r_min, f_min)
    

def global_search_2d(f, params, xb, yb, nx, ny, m = 10000):
    ''' Does a 2D global search using the simulated annealing method.
    xb and yb are tuples defining the search bounds while nx and ny define
    the number of times to stop in eahc direction. The number of minimisations
    done will be N = nx*ny. 
    '''
    
    # Commens have been omitted from the global search 2D and 3D functions
    # as their structure is identitcal to that of the gradient method 
    # global search. The methods should probably have been combined and just
    # have the option to change minimiser but this was added ad-hoc and I
    # ran out of time...
    
    x_range = np.linspace(*xb, nx)
    y_range = np.linspace(*yb, ny)
    
    f_min_global = np.inf
    r_min_global = None
    path_min_global = None
    paths = []
    i = 1
    
    for x in x_range:
        for y in y_range:
            r0 = np.array([x, y])
            path, r_min, f_min = simulated_annealing(f, r0, params)
            paths.append(path)
            
            if(f_min < f_min_global):
                f_min_global = f_min
                r_min_global = r_min
                path_min_global = path
                print(f'New global minimum found at {r_min}, {round(f_min,3)}')
            
            print(f'[{i}/{nx*ny}] Start: {r0} '
                  + f'Minimum: {r_min}, NLL = {round(f_min, 3)}')
            i += 1
    
    print(f'Done. Global minimum: {r_min_global}, {f_min_global}')
    return(paths, path_min_global, r_min_global)
    

def global_search_3d(f, params, xb, yb, zb, nx, ny, nz, m = 10000):
    ''' 3D analogue of the 2D global search. See it for details. 
    '''
    
    x_range = np.linspace(*xb, nx)
    y_range = np.linspace(*yb, ny)
    z_range = np.linspace(*zb, nz)
    
    f_min_global = np.inf
    r_min_global = None
    i = 1
    
    for x in x_range:
        for y in y_range:
            for z in z_range:
                r0 = np.array([x, y, z])
                path, r_min, f_min = simulated_annealing(f, r0, params)
                
                if(f_min < f_min_global):
                    f_min_global = f_min
                    r_min_global = r_min
                    print(f'New global minimum found at {r_min}, {round(f_min,3)}')
                
                print(f'[{i}/{nx*ny*nz}] Start: {r0} '
                      + f'Minimum: {r_min}, NLL = {round(f_min, 3)}')
                i += 1
    
    print(f'Done. Global minimum: {r_min_global}, {f_min_global}')
    return(r_min_global, f_min_global)