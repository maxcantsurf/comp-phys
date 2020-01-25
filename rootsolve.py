def root_find_secant(f, args, x0, n = 100, accuracy = 1e-8):
    # Comparing to the secant method recursion relation xold is x_i-1, 
    # x_new is x_i+1, and x is just x_i
    i = 0
    xold = x0
    x = 0.1
    xnew = 0.1
    while(i < n):
        xnew = x - f(x, *args)*(x - xold)/(f(x, *args) - f(xold, *args))
        xold = x
        x = xnew
        i += 1
        if(abs(xnew - xold) < accuracy):
            print(f'Root finding done: Reached required accuracy of '
                  + f'{accuracy} after {i} iterations')
            return(x)
    print(f'Root finding done: Reached max number of iterations ({n})')
    return(x)