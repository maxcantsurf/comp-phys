===============================================================================
Project 1 Code Submission -- Max Hart -- Computational Physics 2019
===============================================================================
Hello and welcome!
===============================================================================

This is a general overview of how the program works and how to obtain the 
results shown in the report. For a detailed description of how to use the
functions, please see their respective docstrings. 

The code is written according to the PEP-8 style convention (ish). Throughout
the code the quantities theta_23, delta-mass-squared_23 and alpha are referred
to as theta, dmsq, and alpha respectively. These are also commonly shortened to
x, y and z respectively to make the code more readable when neceessary. 

Python was chosen over C++ as the small data set size/low dimensionality means 
that the performance gain is less important than the greater readability 
offered by Python. 

Here is a brief description of the commands needed to reporoduce the results
seen in the report. It is recommended to execute them in the order stated for
things to make the most sense.

===============================================================================
0 -- Setup:
===============================================================================

Before starting make sure main.py is loaded. 

===============================================================================
1 -- Plotting the Data:
===============================================================================

To plot the data use plot_data(). This will produce a plot of the measured
number of events in each energy bin, along with the predicted number of mean
events obtained dusing the fitted parameters. It also shows the resampled
data set, which is explained later on.

===============================================================================
2 -- Minimisations:
===============================================================================

2.1 -- 1D Minimisation:

Use minimise_1d() to carry out minimisation in which theta is varied while
dmsq is held fixed. Cross section energy dependence is neglected. This uses
the parabolic method to find the minimum and the secant method to calculate the
uncertainty. 

2.2 -- 2D Minimisation:

Use minimise_2d() to carry out a minimisation in which both theta and dmsq are
minimised simultaneously. Here the cross section energy dependence is again
ignored. Three different minimisation methods are applied here: the univariate
method, the gradient descent method, the Newton method, simulated annealing and
the quasi-Newton DFP method. The paths taken by each of these minimisers is 
then shown on  a 2D contour plot. Simmulated annealing is also done.

2.3 -- 3D Minimisation:

Use minimise_3d() to carry out the most physically accurate minimisation in
which theta, dmsq and alpha are minimised simultaneously (i.e. the linear
scaling of the cross section with energy is included). 
Here the gradient method, Newton method, simmulated annealing and quasi-Newton 
DFP methods are used.
Their paths are then plotted on three 2D contour plots which represent slices
of the 3D parameter space about a pre-specified minimum which is found
beforehand using the gradient method set to a high degree of accuracy. 


2.4 -- Minimisation Testing:

The functions test_minimisers_2d() and test_minimisers_3d() test the various
minimisation routines on the 2D and 3D Rosenbrock functions respectively. The
minima of these functions is (1, 1) and (1, 1, 1) respectively. 

2.5 -- Global Minimisation Seaches

Using global_min_2d() and global_min_3d() will do minimisations at a regular
grid in the space you are working in. This allows you to search large areas
of the parameter space to check you have indeed found the global minimum.

The 3D method also has a Monte Carlo/simulatex annealing analogue which can be
envolked using mc_global_min_3d().


===============================================================================
3 -- Resampling
===============================================================================

We refer to resampling as the process in which the best parameter estimates,
which we refer to as the 'source parameters' are
used to generate new data sets. The same parameter estimation procedure is then
carried out on this new data set which gives new 'resampled' parameters which
are then compared against the source parameters. This step says nothing of the
physical accuracy of our procedure but rather provides a validation and self
consistency check for the process as a whole.  

To increase the resampled parameter's sensitivity to the source parameters we
can introduce a number of averaging steps. 

3.1 -- Carrying out Resampling

To carry out resampling use resample_singlethread() or resample_multithread().
The latter is just a multi-threaded version of the former. Since resampling
can be CPU intensive the latter command is normally the main one used and so
it is the only one which will actually save data.

*Important*: When using the multi-threaded resampling method, if you are on 
Windows, make sure that you execute the command in an external system console,
otherwise you will not be able to see update messages coming in from other 
threads. 

3.2 -- Viewing Resampling Results

To view the results of resampling, use plot_resample(). This will prompt you
to select the files you wish to include. Note you can select multiple files
and they will be merged together. 

For example, to see the resampling results with no averaging, when the dialog
box opens up, go back and select the files in the folder 'avg1'.

3.3 -- Resampling with Averaging

To view how the resampled data varies with the number of averaging steps used
you can use plot_resample_var_avg(). This will by default plot all of the 
data saved and assumes that it was all generated from the same source 
parameters.

===============================================================================


