import numpy as np
import linear_algebra as la
import uncertainty as uc
import time

def parabolic_1d(f, domain, params, n = 1000, accuracy = 1e-8, 
                        notify = False, error_estimate = False):
    '''Returns the minimum (x, f(x)) of a 1D function via the parabolic method.
    
    Keyword arguments:
        
    f        -- the 1D function f(x, *params) to be minimised
    domain   -- the domain [a, b] over which the function should be minimised, 
                should be in the form of a list [a, b] where a < b
    params   -- a tuple of additional arguments to be passed to the function,
                will be passed in the format f(x, *args)
    n        -- the maximum number of iterations the minimiser should make
                before exiting and returning the current best estimate,
                regardless of whether the desired accuracy is achieved
    accuracy -- if the difference between two successive estimates of the
                minimum x value becomes less than this the routine will exit
                and return the current estimate
    '''
    # 
    xi = np.array([domain[0], (domain[0] + domain[1])/2, domain[1]])
    #x0_old, x1_old, x2_old = domain[0], (domain[0]+domain[1]/2), domain[1]
    y_old = np.inf
    i = 0
    while(i < n):
        x0, x1, x2 = xi[0], xi[1], xi[2]
        y0, y1, y2 = f(xi[0], *params), f(xi[1], *params), f(xi[2], *params)
        x3 = (x2*x2-x1*x1)*y0 + (x0*x0-x2*x2)*y1 + (x1*x1-x0*x0)*y2
        x3 = x3/(2*((x2-x1)*y0 + (x0-x2)*y1 + (x1-x0)*y2))
        y3 = f(x3, *params)
        
        if(abs(y_old - y3) < accuracy):
            if(notify):
                print(f'1D parabolic minimiser done:'
                      + f'Reached required accuracy ({accuracy})')
            if(error_estimate):
                x0 = x1 - 0.02
                x2 = x1 + 0.02
                y0 = f(x0, *params)
                y2 = f(x2, *params)
                xerr = uc.parabolic_error(x3, x0, x1, x2, y0, y1, y2)
                print(f'Error estimate using parabolic curvature = {xerr}')
            return(x3)
            
        if(type(x3) != np.float64):
            if(notify):
                print('1D parabolic minimiser done: Encountered x3 being NaN')
            return((x0+x2)/2, f((x0+x2)/2, *params))
            
        if(y0 > y1 and y0 > y2):
            xi = np.array([x1, x3, x2])
        elif(y2 > y1 and y2 > y0):
            xi = np.array([x0, x3, x1])
        else:
            xi = np.array([x0, x3, x2])
            
        y_old = y3
        
        i += 1
    
    if(notify):
        print(f'Minimisation done: Reached max number of iterations ({n})')
        
    return(x3)
    

def univariate_2d(fx, fy, r0, params, h = 1e-4, n = 10, 
                  accuracy = 1e-8, notify = True):
    x = r0[0]
    y = r0[1]
    path = []
    t0 = time.time()
    reached_accuracy = False
    i = 0
    
    while(i < n):
        rs = np.array([x, y])
        path.append([x, y])
        xparams = y, *params
        x = parabolic_1d(fx, np.array([x-h, x+h]), xparams, accuracy = accuracy)
        
        path.append([x, y])
        yparams = x, *params
        y = parabolic_1d(fy, np.array([y-h, y+h]), yparams, accuracy = accuracy)
        
        if(np.linalg.norm(rs - np.array(x, y)) < accuracy):
            reached_accuracy = True
            break
        
        i += 1
        
    t1 = time.time()
    
    if(notify):
        msg = f'2D parabolic minimiser done: '
        if(reached_accuracy):
            msg += (f'Reached required accuracy of {accuracy} after '
                  + f'{i} iterations')
        else:
            msg += f'Completed max iterations ({n})'
        print(msg)
        print(f'Time taken: {1000*(t1-t0)}ms ({1000*(t1-t0)/i}s per loop)')
        
    return(np.transpose(np.array(path)))
    

def nabla_2d(f, p, r, h):
    ''' Returns an estimate of the gradient at a position vector r for a 2D 
    function with parameters p using a central finite difference scheme with 
    stepping vector h.
    '''
    
    x, y = r[0], r[1]
    hx, hy = h[0], h[1]
    
    # Here we are just using a centra finite difference scheme
    
    dfdx = (f(x+hx,y,*p)-f(x-hx,y,*p))/(2*hx)
    dfdy = (f(x,y+hy,*p)-f(x,y-hy,*p))/(2*hy)
    
    return(np.array([dfdx, dfdy]))
    

def nabla_3d(f, p, r, h):
    ''' 3D version of nabla_2d'''
    
    x, y, z = r[0], r[1], r[2]
    hx, hy, hz = h[0], h[1], h[2]
    
    dfdx = (f(x+hx,y,z,*p)-f(x-hx,y,z,*p))/(2*hx)
    dfdy = (f(x,y+hy,z,*p)-f(x,y-hy,z,*p))/(2*hy)
    dfdz = (f(x,y,z+hz,*p)-f(x,y,z-hz,*p))/(2*hz)
    
    return(np.array([dfdx, dfdy, dfdz]))
    
    
def nabla_4d(f, p, r, h):
    ''' 3D version of nabla_2d'''
    
    x, y, z, w = r[0], r[1], r[2], r[3]
    hx, hy, hz, hw = h[0], h[1], h[2], h[3]
    
    dfdx = (f(x+hx,y,z,w,*p)-f(x-hx,y,z,w,*p))/(2*hx)
    dfdy = (f(x,y+hy,z,w,*p)-f(x,y-hy,z,w,*p))/(2*hy)
    dfdz = (f(x,y,z+hz,w,*p)-f(x,y,z-hz,w,*p))/(2*hz)
    dfdw = (f(x,y,z,w+hw,*p)-f(x,y,z,w-hw,*p))/(2*hw)
    
    return(np.array([dfdx, dfdy, dfdz, dfdw]))
    
    
def hessian_2d(f, p, r, h):
    ''' Returns an estimate of the Hessian matrix at a position vector r for a 
    2D function with parameters p using a central finite difference scheme with 
    stepping vector h.
    '''
    
    # Unpack the position vector r and the stepping vetor h
    
    x, y = r[0], r[1]
    hx, hy = h[0], h[1]
    
    # By calculating the values in this way we save doing extra computations
    
    f1 = f(x+hx, y, *p)
    f2 = f(x-hx, y, *p)
    f3 = f(x, y+hy, *p)
    f4 = f(x, y-hy, *p)
    f5 = f(x, y, *p)
    
    # The values below are the elements of the Hessian matrix, all values
    # are calculated using finite difference centtral difference schemes
    
    d2fdx2 = (f1-2*f5+f2)/(hx*hx)
    d2fdy2 = (f3-2*f5+f4)/(hy*hy)
    d2fdxdy = (f(x+hx,y+hy,*p)-f1-f2+2*f5-f3-f4+f(x-hx,y-hy,*p))/(2*hx*hy)
    H = np.array([[d2fdx2, d2fdxdy], 
                  [d2fdxdy, d2fdy2]])
    return(H)
    
def hessian_3d(f, p, r, h):
    ''' Returns an estimate of the Hessian matrix at a position vector r for a 
    3D function with parameters p using a central finite difference scheme with 
    stepping vector h.
    '''
    
    # Unpack the position vector r and the stepping vetor h
    
    x, y, z = r[0], r[1], r[2]
    hx, hy, hz = h[0], h[1], h[2]
    
    # More central finite difference schemes
    
    H11 = (-f(x+2*hx,y,z,*p)+16*f(x+hx,y,z,*p)-30*f(x,y,z,*p)
            +16*f(x-hx,y,z,*p)-f(x-2*hx,y,z,*p))/(12*hx*hx)
    H22 = (-f(x,y+2*hy,z,*p)+16*f(x,y+hy,z,*p)-30*f(x,y,z,*p)
            +16*f(x,y-hy,z,*p)-f(x,y-2*hy,z,*p))/(12*hy*hy)
    H33 = (-f(x,y,z+2*hz,*p)+16*f(x,y,z+hz,*p)-30*f(x,y,z,*p)
            +16*f(x,y,z-hz,*p)-f(x,y,z-2*hz,*p))/(12*hz*hz)
    
    H12 = (f(x+hx,y+hy,z,*p)-f(x+hx,y-hy,z,*p)-f(x-hx,y+hz,z,*p)+
               f(x-hx,y-hy,z,*p))/(4*hx*hy)
    H13 = (f(x+hx,y,z+hz,*p)-f(x+hx,y,z-hz,*p)-f(x-hx,y,z+hz,*p)+
               f(x-hx,y,z-hz,*p))/(4*hx*hz)
    H23 = (f(x,y+hy,z+hz,*p)-f(x,y+hy,z-hy,*p)-f(x,y-hy,z+hz,*p)+
               f(x,y-hy,z-hz,*p))/(4*hy*hz)
    
    H = np.array([[H11, H12, H13],
                 [H12, H22, H23],
                 [H13, H23, H33]])
    return(H)
    
    
def gradient_method(f, r0, params, n = 1000, alpha = 1e-4, h = 1e-8, 
                    accuracy = 1e-8, notify = True):
    ''' Uses the gradient method to find a minimum a given function. It is not
    dimension-specific. The number of dimensions is specified via the dimension
    of the starting point r0.
    
    Keyword arguments:
        f -- The function to be minimised.
        r0 -- The starting point for the minimiser in the form of a numpy.
        array. Is also used to specify the number of dimensions we are working
        in.
        params -- The additional parameters of the function.
        n -- Maximum number of iterations before exiting.
        alpha -- The step size of the iteration.
        h -- The step size for the finite difference methods used to 
        approximate the gradient.
        accuracy -- Once the current point changes by an amount less than this,
        the algorithm will exit and return the current point.
        notify -- If set to true information regarding the minimisation process
        will be printed upon completion.
    
    Returns:
        A 2D array which contains arrays which hold the coordinate of each 
        point of the path taken by the minimiser, eg: path = [[x0, x1, x2, ...,
        xn], [y0, y1, y2, ..., yn]] for the 2D case. Along with this it also 
        returns a boolean specifying whether the minimiser reached the 
        specified degree of accuracy or not.
    '''
    
    # We start the path off at the given starting point, and use the
    # length of the starting point to define the number of dimensions we 
    # will be working in. We also record the starting system time to get an
    # estimate for the speed per loop of the algorithm
    
    r = r0
    path = []
    dimensions = len(r0)
    h = h*np.ones(dimensions)
    reached_accuracy = False
    t0 = time.time()
    i = 0
    while(i < n):
        r_last = r
        path.append(r)
        grad_f = None
        if(dimensions == 2):
            grad_f = nabla_2d(f, params, r, h)
        elif(dimensions == 3):
            grad_f = nabla_3d(f, params, r, h)
        elif(dimensions == 4):
            grad_f = nabla_4d(f, params, r, h)
        
        r = r - alpha*grad_f
        
        # The change the in the position will be of the order of the magnitude
        # of the gradient times the step size alpha
        
        if(np.linalg.norm(r-r_last) < accuracy):
            reached_accuracy = True
            break
        i += 1
        
    # If the required accuracy isn't reached the algoritm will keep iterating
    # untill the maximum number of iterations 'n' is reached, after which
    # the path is returned anyway
    
    t1 = time.time()
    
    if(notify):
        msg = f'{dimensions}D gradient minimiser done: '
        if(reached_accuracy):
            msg += (f'Reached required accuracy of {accuracy} after '
                  + f'{i} iterations')
        else:
            msg += f'Completed max iterations ({n})'
        print(msg)
        print(f'Time taken: {1000*(t1-t0)}ms ({1000*(t1-t0)/i}s per loop)')
            
            
    # We have been storing the path as a 2D array containing the position
    # vector at each point. It is more conveinient to have three 1D arrays
    # for each coordinate, so we just perform a transpose        
            
    return(np.transpose(np.array(path)), reached_accuracy)


def newton_method(f, r0, params, n = 1000, alpha = 1e-4, h = 1e-8, 
                    accuracy = 1e-8, notify = True):
    ''' Uses the Newton method to search for a stationary point of the given 
    function. The arguments and ourputs are the same as for the gradient search
    method. The only difference is that the alpha value is now a damping
    factor applied to the step to aid with stability, i.e. the step size is
    alpha*gradient*hessian. 
    '''
    
    r = r0
    path = []
    dimensions = len(r0)
    h = h*np.ones(dimensions)
    reached_accuracy = False
    t0 = time.time()
    i = 0
    while(i < n):
        
        r_last = r
        path.append(r)
        
        grad_f = None
        H = None
        inv_H = None
        
        # We could perhaps make some super-complicated function which 
        # calculates the finite difference approximation of the Hessian
        # for an arbitrary number of dimensions, however here since we are
        # dealing with just up to three dimensions there is little point so
        # we explcitly check the dimensions and hard code the Hessian
        # computation for each dimension
        
        if(dimensions == 2):
            grad_f = nabla_2d(f, params, r, h)
            H = hessian_2d(f, params, r, h)
            
        elif(dimensions == 3):
            grad_f = nabla_3d(f, params, r, h)
            H = hessian_3d(f, params, r, h)
            
        # We invert the hessian using the method of LU decomposition with 
        # forward and back substitution. We can re-use the code from the 
        # assignment 'matrix methods' task
        # After this we finally update the current position vector
            
        inv_H = np.array(la.invert(H))
        r = r - alpha*np.dot(inv_H, grad_f)
        
        if(np.linalg.norm(r-r_last) < accuracy):
            reached_accuracy = True
            break
        i += 1
    t1 = time.time()
        
    if(notify):
        msg = f'{dimensions}D Newton minimiser done: '
        if(reached_accuracy):
            msg += (f'Reached required accuracy of {accuracy} after '
                  + f'{i} iterations')
        else:
            msg += f'Completed max iterations ({n})'
        print(msg)
        print(f'Time taken: {1000*(t1-t0)}ms ({1000*(t1-t0)/i}ms per loop)')
            
    return(np.transpose(np.array(path)), reached_accuracy)

    
def quasi_newton(f, r0, params, n = 1000, alpha = 1e-4, h = 1e-8, 
                    accuracy = 1e-8, notify = True, update_method = 'dfp'):
    ''' Uses a specified quasi-Newton update method to search for the 
    stationary point of a given function. The arguments, output and algorithm
    structure are the same as the gradient search algorithm. The additional
    keyword argument 'update_method' specifies which update method to use.
    
    Possible options for the update method are:
        dfp -- Davidson-Fletcher-Powell
        bfgs  -- Broyden–Fletcher–Goldfarb–Shanno 
        broyden -- Broyden method
        sr1 -- Symmetric-Rank 1 
    '''
    
    # The structure of the algorithm is the same as the Gradient and Newton 
    # method. The only thing which really changes is the update matrix used
    
    r = r0
    p = params
    dimensions = len(r0)
    h = h*np.ones(dimensions)
    path = []
    reached_accuracy = False
    G = np.identity(dimensions)
    t0 = time.time()
    i = 0
    
    while(i < n): 
        r_last = r
        path.append(r)
        
        if(dimensions == 2):
            grad_f = nabla_2d(f, p, r, h)
            r_new = r - alpha*np.dot(G, grad_f)
            grad_f_new = nabla_2d(f, p, r_new, h)
            
        elif(dimensions == 3):
            grad_f = nabla_3d(f, p, r, h)
            r_new = r - alpha*np.dot(G, grad_f)
            grad_f_new = nabla_3d(f, p, r_new, h)
            
        gamma = grad_f_new - grad_f
        delta = r_new - r
        delta_outer = np.outer(delta, delta)
        
        # Here we set the update matrix according to the algorithm the user
        # has specified. 
        
        if(update_method == 'dfp'):
            A = delta_outer/np.dot(gamma, delta) 
            B = np.dot(G,np.dot(delta_outer,G))/np.dot(gamma,np.dot(G,gamma))
            G = G + A - B
            
        elif(update_method == 'bfgs'):
            I = np.identity(dimensions)
            inner_gamma_delta = np.dot(gamma, delta)
            A = I - np.outer(delta, gamma)/inner_gamma_delta
            B = I - np.outer(gamma, delta)/inner_gamma_delta
            C = np.outer(delta, delta)/inner_gamma_delta
            G = np.dot(A, np.dot(G, B)) + C
            
        elif(update_method == 'broyden'):
            A = np.outer(delta - np.dot(G, gamma), delta)
            B = np.dot(gamma, np.dot(G, gamma))
            G = G + np.dot(A, G)/B
        
        elif(update_method == 'sr1'):
            A = delta - np.dot(G, gamma)
            G = G + np.outer(A, A)/np.dot(A, gamma)
            
        else:
            raise(ValueError('Invalid update method given'))
            return
        
        r = r_new
        
        if(np.linalg.norm(r-r_last) < accuracy):
            reached_accuracy = True
            break
        i += 1
    t1 = time.time()
        
    if(notify):
        msg = f'{dimensions}D quasi-Newton ({update_method}) minimiser done: '
        if(reached_accuracy):
            msg += (f'Reached required accuracy of {accuracy} after '
                  + f'{i} iterations')
        else:
            msg += f'Completed max iterations ({n})'
        print(msg)
        print(f'Time taken: {1000*(t1-t0)}ms ({1000*(t1-t0)/i}ms per loop)')
        
    return(np.transpose(np.array(path)), reached_accuracy)   
    

def global_gradient_3d(f, x_domain, y_domain, z_domain, nx, ny, nz, params,
                         minimiser_params):
    ''' Searches for the global minimum by carrying out the gradient method
    at regular intervals over the given domain. The x, y and z domains are
    tuples which define the lower and upper limits, while the nx, ny and nz
    values are integers which define the number of times we should stop to
    minimise in each direction. The total number of minimisations is then
    N = nx*ny*nz
    '''
    
    # Contruct the arrays we will iterate over
    
    x_arr = np.linspace(*x_domain, nx)
    y_arr = np.linspace(*y_domain, ny)
    z_arr = np.linspace(*z_domain, nz)
    
    # The variables below will hold our current best minimum
    
    path_global = None
    f_min_global = np.inf
    reached_accuracy_global = False
    
    # Now we just loop over the 3D grid we have greated, using each point
    # as a different starting point for our gradient minimiser
    
    for x in x_arr:
        for y in y_arr:
            for z in z_arr:
                
                r0 = np.array([x, y, z])
                path, ra = gradient_method(f, r0, params, *minimiser_params,
                        notify = True)
                
                x_min = path[0][-1]
                y_min = path[1][-1]
                z_min = path[2][-1] 
                
                f_start = f(x, y, z, *params)
                f_min = f(x_min, y_min, z_min, *params)
                
                r1 = np.array([x_min, y_min, z_min])
                
                print(f'Start point: {np.round(r0, 3)}, nll = {round(f_start, 4)}')
                print(f'End point:   {np.round(r1, 3)}, nll = {round(f_min, 4)}')
                
                # If the minimum we found has an NLL less than the current
                # lowest NLL, it will become the new best minimum
                
                if(f_min < f_min_global):
                    path_global = path
                    f_min_global = f_min
                    reached_accuracy_global = ra
                    
    return(path_global, reached_accuracy_global)
    