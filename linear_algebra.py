def print_matrix(matrix, precision=3):
    ''' Prints a given matrix in a 'pretty' fashion.

    Keyword arguments:
    matrix - the matrix to be printed, must be given in the standard matrix
    format:
    [[11, 12, ... , 1n]
     [21, 22, ... , 2n]
     .    .   .     .
     .    .      .  .
     [m1, m2, ... , mn]]
    precision - the number of decimal places to print the elements to
    '''

    # Don't have to worry about indices etc here so will just use for loops
    # to iterate over all elements of the matrix
    # In the loops below we are just finding the maximum number of digits
    # needed to represent all of the numbers rounded to the given precision
    # This is so we can amke each row have the same number of characters
    # (Which looks pretty)

    maxlength = 0
    for row in matrix:
        for element in row:
            length = len(str(round(element, precision)))
            if(length > maxlength):
                maxlength = length

    # Now we know the maxlength we again go through all the elements and
    # construct a new matrix whose rows will all have the same number of
    # characters

    for row in matrix:
        newrow = []
        for element in row:
            length = len(str(round(element, precision)))
            # If the element length is smaller than the max length, in order
            # for elements to all have the same length we will need to pad
            # the smaller ones with whitespace
            if(length != maxlength):
                # The amount of whitespace we need is the difference between
                # the element length and the maxlength
                whitespace = ' '*(maxlength - length)
                newrow.append((whitespace + str(round(element, precision))))
            else:
                # If the element has length equal to maxlength we don't need
                # to do anything
                newrow.append(str(round(element, precision)))
        print('[' + ','.join(newrow) + ']')


def lu_decomp(A):
    N = len(A)
    # Checking that the input matrix is square
    if(N != len(A[0])):
        raise(TypeError('Matrix must be square'))
        return(None)
    # Create square matrices for L and U filled with zeros
    L = [[0 for i in range(N)] for j in range(N)]
    U = [[0 for i in range(N)] for j in range(N)]
    i = 1
    # Set the diagonal elements of L to unity
    while(i <= N):
        L[i-1][i-1] = 1
        i += 1
    j = 1
    # Begin the actual decomposition
    while(j <= N):
        # We first compute the elements of the upper matrix by carrying out
        # matrix multiplication in reverse
        i = 1
        while(i <= j):
            s = sum([L[i-1][k-1]*U[k-1][j-1] for k in range(1, i)])
            U[i-1][j-1] = A[i-1][j-1] - s
            i += 1

        i = j + 1
        while(i <= N):
            s = sum([L[i-1][k-1]*U[k-1][j-1] for k in range(1, j)])
            L[i-1][j-1] = (A[i-1][j-1] - s)/U[j-1][j-1]
            i += 1
        j += 1
    # Return L and U as seperate matrices in a tuple
    return(L, U)


def lu_solve(L, U, b):
    ''' Returns the solution vector x to the matrix equation L.U.x = b where
    L and U are upper and lower diagonal matrices respectively.

    Keyword arguments:
    L - a square N x N lower-diagonal matrix
    U - a square N x N upper-diagonal matrix
    b - a vector of length N which represents the RHS for L.U.x = b
    '''
    N = len(L)
    # Check that the RHS input b has the correct dimensions for a vector
    if(N != len(b)):
        raise(TypeError('b must have dimensions N x 1'))
        return(None)
    # Create empty arrays which will hold the xi and yi values
    y = [0 for i in range(N)]
    x = [0 for i in range(N)]
    # We start from i = 2 as the i = 1 case is done before the loop
    # The loop then carries out the forward substitution, i.e. it solves the
    # equation L.y = b for y
    i = 2
    y[0] = b[0]/L[0][0]
    while(i <= N):
        # s = sum([L[i-1][j-1]*y[j-1] for j in range(1, i - 1)])
        s = 0
        j = 1
        while(j <= i-1):
            s += L[i-1][j-1]*y[j-1]
            j += 1
        y[i-1] = (b[i-1] - s)/L[i-1][i-1]
        i += 1
    # Now the yi have been found we carry out a backward substitution to solve
    # the equation U.x = y for x.
    i = N - 1
    x[N-1] = y[N-1]/U[N-1][N-1]
    while(i >= 1):
        # s = sum([U[i-1][j-1]*x[j-1] for j in range(i + 1, N)])
        s = 0
        j = i + 1
        while(j <= N):
            s += U[i-1][j-1]*x[j-1]
            j += 1
        x[i-1] = (y[i-1] - s)/U[i-1][i-1]
        i -= 1
    # With the backward substitution done we now just return the vactor x
    # which is the solution to the full equation A.x = L.U.x = L.y = b
    return(x)


def multiply(A, B):
    ''' Returns the product of the matrices A and B.'''

    n = len(A)
    m = len(A[0])
    p = len(B[0])

    if(m != len(B)):
        print('Matrices must be n x m and m x p')
        return(None)

    # This is just a straightforward explicit implementation of the definiton
    # of matrix multiplication. Here C holds the value of the product, which
    # will be an nxp matrix

    C = [[0 for i in range(p)] for j in range(n)]
    i = 1
    while(i <= n):
        j = 1
        while(j <= p):
            s = 0
            k = 1
            while(k <= m):
                s = s + A[i-1][k-1]*B[k-1][j-1]
                k += 1
            C[i-1][j-1] = s
            j += 1
        i += 1
    return(C)


def determinant(A):
    ''' Returns the determinant of A using LU decomposition.'''
    L, U = lu_decomp(A)
    det = 1
    i = 1
    # The determinant is simply the product of all of the diagonal elements of
    # the upper matrix, so we just iterate over all of the diagonal elements
    # U[i][i] and return the product
    while(i <= len(A)):
        det = det*U[i-1][i-1]
        i += 1
    return(det)


def invert(A):
    ''' Returns the inverse of a given matrix A using LU decomposition. '''

    # First we decompose A into L and U
    L, U = lu_decomp(A)
    N = len(A)
    i = 0
    inverse = []
    while(i < N):
        # Here the 'basis' is the ith column vector of the identity matrix
        basis = [0 for i in range(N)]
        basis[i] = 1
        # For each column vector of the identity matrix we solve the matrix
        # equation A.basis = x to give the ith column vector of the inverse
        # of A, which we then append to the list which will hold all these
        inverse.append(lu_solve(L, U, basis))
        i += 1
        # We have been adding the column vectors to 'inverse' as 1D lists
        # In our convention, the lists form rows, not columns, thus we have
        # to do a transpose before returning
    return(transpose(inverse))


def transpose(A):
    ''' Returns the transpose of a given matrix A. '''

    # We first check that the given matrix A is square
    N = len(A)
    if(N != len(A[0])):
        raise(TypeError('Matrix must be square'))
        return(None)
    # Now we know the matrix is square we can create an empty square NxN matrix
    # which will be the transposed matrix.
    matrix = [[0 for i in range(N)] for j in range(N)]
    # We now populate the transposed matrix by iterating over all of the
    # elements of the input matrix. This is simple as the element ij of the
    # transposed matrix is A[j][i] by definition
    i = 0
    while(i < N):
        j = 0
        while(j < N):
            # Notice how we are just swapping the indices
            matrix[i][j] = A[j][i]
            j += 1
        i += 1
    return(matrix)

