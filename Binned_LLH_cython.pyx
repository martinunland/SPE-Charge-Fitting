# @file Binned_LLH_cython.pyx
#
#
# Functions and Class for data handling and fitting of SPE spectra with binned log likelihood. Model from Bellamy's paper 
# (https://doi.org/10.1016/0168-9002(94)90183-X) although be aware that eq. 8 is wrong! I use the correct calculation of eq. 7.
#
# @author Martin Unland   m.unland@wwu.de
# @date August 2020
#
import cython
cimport cython
import numpy as np
cimport numpy as np
import math as mt
from libc.math cimport exp, M_PI, sqrt, log, floor, erf, abs, erfc
from scipy.stats import poisson
from scipy.special import gammaln


cpdef double _gaussian(double x, double mean, double sigma):
    return exp(-(x-mean)**(2.)/(2.*sigma**(2.)))/(sigma*sqrt(2.*M_PI))

## 
#   Fit function of SPE spectrum
#       @see Binned_LLH.compute for parameter description 
#       @return Area under function between given bin edges using Riemann sum
cpdef double _fit_function(tuple x, double Q0, double s0, double Q1,
                                   double s1, double poisson_mean, double height, double n_omega,
                                   double n_alpha):
    n_alpha = 1./n_alpha
    cdef double temp = 0.0
    cdef int nr_of_gaussians = 30 #Maximal number of MPE
    cdef double temp2 = 0.0
    cdef double x1 = x[0]
    cdef double x2 = x[1]
    cdef total_poisson_calc = 0.0
    cdef poisson_prob = 0.
    cdef mean
    cdef sigma
    for i in range(0, nr_of_gaussians):
        mean = Q0 + Q1*i
        sigma = sqrt(s0**2+i*s1**2)
        poisson_prob = _poisson(poisson_mean, i)
        temp += height*poisson_prob*((1.-n_omega)*_gaussian(x1, mean, sigma)+
                                           n_omega*_background_eq8(x1,Q0,mean,sqrt(i)*s1,n_alpha)) 
        temp2 += height*poisson_prob*((1.-n_omega)*_gaussian(x2, mean, sigma)+
                                            n_omega*_background_eq8(x2,Q0,mean,sqrt(i)*s1,n_alpha))
        total_poisson_calc += poisson_prob
        
        if total_poisson_calc > 0.9999: #Break sum when 99.99% of poisson distribution is accounted for
            break
    return (temp2+temp)*(x2-x1)/2.

cpdef double step(double x):
    if x>=0:
        return 1.
    else:
        return 0.

##
#   Equation (8) in Bellamy et al paper, but calculated correctly. 
#   Used by _fit_function and _integral_fit_function.
#   Convolution between gaussian and exponential = Exponentially modified Gaussian Distribution.
#       @see Binned_LLH.compute for parameter description 
cpdef double _background_eq8(double x, double Q0, double Qn, double sn, double n_alpha):
    cdef double coef = Qn+n_alpha*sn**2./2.
    cdef double coef2 = Qn+n_alpha*sn**2.
    cdef double div = (sn*sqrt(2.))
    if sn>0.:
        return (n_alpha/2.)*exp(-n_alpha*(x-coef))*(erfc((coef2-x)/div))

    else:
        return n_alpha*step(x-Q0)*exp(-n_alpha*(x-Q0))

cpdef double _poisson(double mean, int n):
    
        return exp(-mean)*mean**n/mt.factorial(n)#

## 
#   Fit function of SPE/MPE distributions, which can be used for plotting
#       @see Binned_LLH.compute for parameter description 
#       @N_start : first S/M PE peak
#       @N_stop  : stopping at this S/M PE peak
#       E.g. you want to plot only 2-Peak -> N_start=2, N_stop=3
#       @return Area under function between given bin edges using Riemann sum (I've used quad for numerical integration, but it takes a lot longer and results do not change much (do not use too wide bins though!)
cpdef double fit_func_for_plot(tuple x, double Q0, double s0, double Q1,
                                double s1, double poisson_mean, double height, double n_omega,
                                double n_alpha, int N_start, int N_stop):
    n_alpha = 1./n_alpha
    cdef double temp = 0.0
    cdef double temp2 = 0.0
    cdef double x1 = x[0]
    cdef double x2 = x[1]
    cdef int i
    for i in range(N_start, N_stop):
        mean = Q0 + Q1*i
        sigma = sqrt(s0**2+i*s1**2)
        temp += height*_poisson(poisson_mean, i)*((1.-n_omega)*_gaussian(x1, mean, sigma)+
                                           n_omega*_background_eq8(x1,Q0,mean,sqrt(i)*s1,n_alpha)) 
        temp2 += height*_poisson(poisson_mean, i)*((1.-n_omega)*_gaussian(x2, mean, sigma)+
                                            n_omega*_background_eq8(x2,Q0,mean,sqrt(i)*s1,n_alpha))
    return (temp2+temp)*(x2-x1)/2.


cpdef double lininter(double x,double x0,double x1,double y0,double y1):
    return y0+(x-x0)*(y1-y0)/(x1-x0)

## 
#   Calculates the Initial parameters for the fit.
#       @param x_data : bin edges of charge spectrum
#       @param y : bin counts of charge spectrum
#       @param Q0     : mean of pedestal
#       @param s0     : st. dev. of pedestal
cpdef object Initial_paramters(x_data, y, Q0,  s0):
    
    xstep = (x_data[1]-x_data[0])*0.5
    x = np.linspace(x_data[0]+xstep, x_data[-1]-xstep, y.size)
    
    x_of_pedestal = np.logical_and(x > Q0 - 2.*s0, x < Q0 + 2.*s0) #Range of 95% pedestal
    counts_pedestal = sum(y[x_of_pedestal])/0.95 #counts of pedestal
    
    mu = -np.log(counts_pedestal/float(sum(y))) #Get mu of poisson from Poisson(0,mu) = exp(-mu) = N_0/N_Total
    average_charge = sum(y*x)/float(sum(y)) 
    
    Q1_0 = (average_charge-Q0)/mu 
    height_0 = float(sum(y))
    
    p0 = {'Q0' : Q0, 's0' :  s0, 'Q1' : Q1_0, 's1' : 0.4*Q1_0, 'poisson_mean' : mu,
          'height' : height_0, 'n_omega' : 0.03, 'n_alpha' : Q1_0/2.}
    return p0 






## 
#   Class for data handling and providing likelihood to be minimized. Based on exampled from https://iminuit.readthedocs.io/en/latest/tutorials.html
#   Takes the SPE spectrum as input. The compute() function was coded for iminuit. If you want to use another minimizer you may have to change stuff.
cdef class Binned_LLH:  
    cdef np.ndarray x_data
    cdef np.ndarray y_data
    cdef double step
    cdef int ndata
    cdef double logfactorial
    
    ##      Class Constructor
    #       @param x_data : bin edges of charge spectrum
    #       @param y_data : bin counts of charge spectrum
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.ndata = len(y_data)
        self.logfactorial = 0
        for i in range(0, self.ndata):
            self.logfactorial = self.logfactorial + gammaln(self.y_data[i]+1)

            
    ##      Compute log likelihood for given model
    #       @param Q0           : mean of pedestal
    #       @param s0           : st. dev. of pedestal
    #       @param Q1           : mean of 1 PE peak
    #       @param s1           : st. dev. of 1 PE peak
    #       @param poisson_mean : mean number of detected photons per trigger
    #       @param height       : total number of counts... you can leave it fixed in iminuit 
    #       @param n_omega      : probability for a background event (s. Bellamy's paper)
    #       @param n_alpha      : average charge of background event (s. Bellamy's paper)
    
    @cython.embedsignature(True)  # you need this to dump function signature in docstring
    cpdef double compute(self, double Q0, double s0, double Q1,
                double s1, double poisson_mean, double height, double n_omega,double n_alpha):
        #this line is a cast not a copy. Let cython knows mydata will spit out double
        cdef np.ndarray my_y_data = self.y_data
        cdef np.ndarray my_x_data = self.x_data
        cdef np.ndarray y_fit_np
        cdef list y_fit = []
        cdef double tmp 
        for i in range(0, self.ndata):
            y_fit.append(_fit_function((my_x_data[i],my_x_data[i+1]), Q0, s0, Q1, s1, poisson_mean,
                                       height,n_omega, n_alpha))
        y_fit_np = np.array(y_fit)
        return -(np.nansum(-y_fit_np+my_y_data*np.log(y_fit_np))- self.logfactorial)
    
     ##     Goodness of fit
     #      GOF for binned data following poisson error using ratio test with data as null hypothesis can be interpreted as red. Chi2 as long as counts are large
     #      see slide 94 https://indico.cern.ch/category/6015/attachments/192/631/Statistics_Fitting_II.pdf
    @cython.embedsignature(True)
    cpdef double goodness_of_fit(self, double Q0, double s0, double Q1,
                double s1, double poisson_mean, double height, double n_omega,double n_alpha):
        cdef np.ndarray my_y_data = self.y_data
        cdef np.ndarray my_x_data = self.x_data
        cdef np.ndarray y_fit_np
        cdef list y_fit = []
        cdef double tmp 
        for i in range(0, self.ndata):
            y_fit.append(_fit_function((my_x_data[i],my_x_data[i+1]), Q0, s0, Q1, s1, poisson_mean,
                                       height,n_omega, n_alpha))
        y_fit_np = np.array(y_fit)
        return 2*np.nansum((y_fit_np-my_y_data)+my_y_data*np.log(my_y_data/y_fit_np))/float(self.ndata)
     
