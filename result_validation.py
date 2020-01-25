import numpy as np
import random
import minimisation as mn
import base_functions as bf
import multiprocessing as mp
import time
import os

def poisson(lmda):
    ''' Generates a random sample from a Poisson distribution with a given mean
    lmda. This algorithm is the work of Donald Knuth, from his book:
    'The Art of Computer Programming', found at
    https://www-cs-faculty.stanford.edu/~knuth/taocp.html
    '''
    
    L = np.exp(-lmda)
    k = 0
    p = 1
    
    while(p > L):
        k = k + 1
        p = p*random.uniform(0, 1)
        
    return(k - 1)


def generate_data(theta, dmsq, alpha, flux, energy, length):
    ''' Generates a new data set using given values for theta, dmsq and
    alpha. To generate a data set which does not include cross section energy
    dependence take alpha = np.reciprocal(energy)
    '''
    
    # First create a list to store the energies
    # We then iterate through the each energy bin and its corresponding flux
    # For each bin we use the given parameters along with the energy and flux
    # to generate a new random value of lamda using the Poisson generator
    # This is done for the whole energy range, after which the data is returned
    
    data = []
    for f, e in zip(flux, energy):
        data.append(poisson(bf.lmda(theta, dmsq, alpha, f, e, length)))
        
    return(data)


def resample(theta, dmsq, alpha, flux, energy, length, minimiser_params,
             num_resamples = 10, num_averages = 100):
    ''' Uses the given input parameters to generate new data sets and then
    carry out parameter estimation on these new data sets.
    
    Keyword arguments:
        minimiser params -- A tuple containing the parameters for the 
            minimisation algorithm to use. Typically of the form
            (n, alpha, h, accuracy, notify)
        num_resamples -- The number of minimisations to be carried out/
            number of sets of parameters to be returned
        num_averages -- The number of data sets which will be generated
            to produce an average data set which is then fed into the 
            minimiser. To disable data set averaging set this to one.
            
    Returns:
        A tuple containing lists of the parameters and the corresponding nll
        for the resampled data, of the form (theta, dmsq, alpha, nll)
    '''
    
    # Getting the process name is useful for when we are running this in 
    # multithreaded mode and want to know the progress of each thread
    # individually
    
    pid = mp.current_process().name
    x_mins = []
    y_mins = []
    z_mins = []
    nll_mins = []
    i = 1
    
    while(i <= num_resamples):
        
        # lmda_average will store the running averaged data set
        # We then perform a loop in which we generate data sets and perform
        # a running average. Performing a running average as opposed to
        # averaging at the end results in reduced memory consumption
        
        lmda_average = np.zeros(len(energy))
        j = 1
        while(j <= num_averages):
            # Generate a new data set
            new_data = np.array(generate_data(theta, dmsq, alpha, flux,
                                                    energy, length))
            # Add the newly generated data set to the running average
            lmda_average = lmda_average*(j-1)/j + new_data/j
            j += 1
            
        # Now we actually carry out the minimisation/parameter estimation
        # using our generated averaged data set. An obvious choice of starting
        # point is the source parameters which were used to generate the 
        # new data sets

        params = (lmda_average, flux, energy, length)
        r0 = np.array([theta, dmsq, alpha])
        path, ra = mn.gradient_method(bf.nll, r0, params, *minimiser_params)
        
        # We only include the run if the minimisation managed to reach the
        # specified accuracy, otherwise we might enp up including 
        # runs which did not actually find the minimum
        
        if(ra):
            # The optimum parameters are those at the end of the minimisation
            # path, we also calculate the NLL for good measure
            
            x_min = path[0][-1]
            y_min = path[1][-1]
            z_min = path[2][-1]
            
            nll_min = bf.nll(x_min, y_min, z_min, *params)
            
            x_mins.append(x_min)
            y_mins.append(y_min)
            z_mins.append(z_min)
            nll_mins.append(nll_min)
            
            # This just helps us keep track of the progress each thread is
            # making
            
            print(f'[Thread {pid}]:({i}/{num_resamples}): theta = {x_min}, '
                  + f'dmsq = {y_min}, alpha = {z_min}, nll = {nll_min}')
            
            i += 1
            
    return(x_mins, y_mins, z_mins, nll_mins)
    

def resample_multithread(theta, dmsq, alpha, flux, energy, length, 
                         minimiser_params, num_threads = 10, m = 10,
                         num_averages = 100):
    ''' Completes the same task as the normal resample method except it 
    uses the multiprocessing module to take advantage of multiple cores/threads
    on a machine. 
    
    Keyword arguments:
        num_threads -- The number of processes to spawn and execute in 
        parallel. For optimal results this should be set to be the number of
        threads your processor possess, or one less than this to maintain
        operating system performance.
        m -- The number of resamples to perform per process. This means that a
        total of num_threads*m minimisations will be performed.
        num_averages -- The number of data sets to be produced and averged
        over ot produce the averaged data set. Setting this to one will
        result in no averaging.
        
    Returns:
        A tuple containing lists of the parameters and the corresponding nll
        for the resampled data, of the form (theta, dmsq, alpha, nll)
        It will also save this returned data to a text file under 
        /resample_data/ so it can be analysed later on.
    '''
    
    # We first create a pool which will contain and handle the processes which
    # will execute our tasks in parallel. The loop after this is just
    # preparing the arguemnts for the functions which will be executed in
    # parallel
    
    p = mp.Pool()
    args = [(theta, dmsq, alpha, flux, energy, length, minimiser_params, m,
             num_averages) for i in range(num_threads)]
    
    # The starmap function executes the task pool. It uses the number of 
    # sets of arguments you provide to know how many processes to spawn.
    # Once all of these processes are done it returns a nicely formatted
    # list of the return values of each of the functions
    
    results = p.starmap(resample, args)
    
    x_list = []
    y_list = []
    z_list = []
    nll_list = []
    
    # In the loop below we are unpacking the result from each process and
    # 'stiching' them rogether to form lists for each of the parameters
    
    for result in results:
        x_sublist, y_sublist, z_sublist, nll_sublist = result
        x_list += x_sublist
        y_list += y_sublist
        z_list += z_sublist
        nll_list += nll_sublist
        
    # These computations take a long time so it is a good idea to save the
    # result so the data can be analysed later on
    
    directory = os.fsencode(os.getcwd() + f'/resample_data/avg{num_averages}')
    
    if not os.path.exists(directory):
        os.makedirs(directory)    
    
    np.savetxt(f'resample_data/avg{num_averages}/n{num_threads}_m{m}_avg{num_averages}_'
               + f'theta{theta}_dmsq{dmsq}_alpha{alpha}_'
               + time.strftime("%Y%m%d-%H%M%S"), 
               (x_list, y_list, z_list, nll_list))
    
    # Although this method is intended to be called just to save the results
    # it produces, we also return the results if we want to analyse the 
    # output straight away
    
    return(x_list, y_list, z_list, nll_list)
    
    
def eigenvalues3x3(A):
    ''' Computes and returns the eigenvalues of the given 3x3 matrix A using an 
    explicit method.
    '''
    
    # This algorithm was adapted from 
    # https://dl.acm.org/citation.cfm?doid=355578.366316
    
    # We first check if the matrix is triangular, if so the eigenvalues are
    # just the diagonal elements
    
    p1 = A[0][1]**2 + A[0][2]**2 + A[1][2]**2
    if(p1 == 0) :
       e1 = A[0][0]
       e2 = A[1][1]
       e3 = A[2][2]
      
    # The rest of the algorithm is just taken directly from the source above
    # where the workings of the algorithm are explained in detail
       
    else:
       q = (A[0][0] + A[1][1] + A[2][2])/3
       p2 = (A[0][0]-q)**2 + (A[1][1]-q)**2 + (A[2][2]-q)**2 + 2*p1
       p = np.sqrt(p2/6)
       B = (1/p)*(A-q*np.identity(3))   
       r = np.linalg.det(B)/2
    
       if(r <= -1):
          phi = np.pi / 3
       elif(r >= 1):
          phi = 0
       else:
          phi = np.arccos(r)/3
    
       e1 = q + 2*p*np.cos(phi)
       e3 = q + 2*p*np.cos(phi + (2*np.pi/3))
       e2 = 3*q - e1 - e3    
       
       return(e1, e2, e3)
       
       
def check_definiteness_3d(f, r0, params):
    ''' Checks whether the Hessian of a 3D function at a given point is 
    positive definite. If it is then we know the given point is a minimum.
    '''
    
    # First we calculate the hessian
    
    H = mn.hessian_3d(f, params, r0, 1e-6*np.ones(3))
    print('Hessian at end point is H =')
    print(H)
    print('')
    
    # Now we find the eigenvalues of the Hessian we just computed
    
    e1, e2, e3 = eigenvalues3x3(H)
    print(f'Eigenvalues of Hessian are {e1, e2, e3}')
    
    # Now we just work through the different cases. If all the eigenvalues are
    # positive, it is poitive definite => minimum, if tey are all negative,
    # it is negative definite => maximum, otherwise. If neither of these is
    # true we are either at a saddle point or a point which doesn't even have
    # a gradient of zero
    
    if(e1 > 0 and e2 > 0 and e3 > 0):
        print('H is positive definite => r0 is a minimum!')
    elif(e1 < 0 and e2 < 0 and e3 < 0):
        print('H is positive definite => r0 is a maximum!')
    else:
        print('H is indefinite, r0 may be a saddle point!')
    
     
def check_eiegenvalue_method():
    ''' A validation for our eigenvalue-finding algorithm. We apply the method
    to a matrix with known eigenvalues and compare the results.
    '''
    
    A = np.array([[3, 1, 1],
                  [1, 2, 2],
                  [1, 2, 2]])
    print('The symmetric matrix A = ')
    print('')
    print(A)
    print('')
    print('has eigenvalues, 0, 2, 5')
    print('Eigenvalue finding algorithm result:')
    print(eigenvalues3x3(A))

