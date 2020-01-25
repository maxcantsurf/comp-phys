def parabolic_error(x_min, x0, x1, x2, y0, y1, y2):
    ''' Takes three points from a quadratic Largange polynomial, fits them,
    and uses the curvature around the minimum to estimate the error in the
    minimum. 
    '''
    
    d0 = y0/((x0-x1)*(x0-x2))
    d1 = y1/((x1-x0)*(x1-x2))
    d2 = y2/((x2-x0)*(x2-x1))
    
    a = d0+d1+d2
    b = -d0*(x1+x2)-d1*(x0+x2)-d2*(x1+x0)
    
    return(abs(0.5/(2*a*x_min+b)))