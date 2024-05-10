# using Kalousis function of https://arxiv.org/pdf/2304.08735
#
import cython
cimport cython
import numpy as np
cimport numpy as np
import math as mt
from libc.math cimport exp, M_PI, sqrt, log, floor, erf, abs, erfc
from scipy.special import gammaln, hyp1f1, hyperu, gamma

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
    cdef double x1 = x[0]
    cdef double x2 = x[1]
    cdef double temp = _fit_function_single_x(x1, Q0, s0, Q1, s1, poisson_mean, height, n_omega, n_alpha)
    cdef double temp2 = _fit_function_single_x(x2, Q0, s0, Q1, s1, poisson_mean, height, n_omega, n_alpha)
    return (temp2+temp)*(x2-x1)/2.

cpdef double _SR0(double x, double Q0, double s0, double poisson_mean):
    cdef double arg0 = (x - Q0) / s0
    return (1.0 / (sqrt(2.0 * M_PI) * s0)) * exp(-0.5 * arg0 * arg0)

cpdef double _SR1(double x, double Q0, double s0, double Q1,
                                   double s1, double poisson_mean, double height, double n_omega,
                                   double n_alpha):

    cdef double omega0 = (x - Q0 - n_alpha * s0 * s0) / (sqrt(2.0) * s0)
    cdef double SR1 =  n_omega * n_alpha / 2.0 * exp((n_alpha* n_alpha * s0 * s0) / 2.0 - n_alpha * (x - Q0)) * erfc(-omega0)

    cdef double mean = Q0 + Q1
    cdef double sigma = sqrt(s0 * s0 + s1 * s1)
    
    cdef double arg1 = (x - mean) / sigma
    
    cdef double gn = 0.5 * erfc(-Q1 / (sqrt(2.0) * s1))
    cdef double A = (Q0 - x) * s1 * s1 - Q1 * s0 * s0
    cdef double B = sqrt(2.0) * s0 * s1 * sigma
    
    SR1 += ((1.0 - n_omega) / (2.0 * gn * sqrt(2.0 * M_PI) * s1)) * exp(-0.5 * arg1 * arg1) * erfc(A / B)

    return SR1 

cpdef double _SRn(double x, double Q0, double s0, double Q1,
                                   double s1, double poisson_mean, double height, double n_omega,
                                   double n_alpha, int n):
    cdef double gn = 0.5 * erf(-Q1 / (sqrt(2.0) * s1))
    cdef double k = s1 / (gn * sqrt(2.0 * M_PI)) * exp(-Q1 * Q1 / (2.0 * s1 * s1))
    cdef double Qg = Q1 + k
    cdef double sg2 = s1 * s1 - Qg * k

    cdef double Qn, sn2, sn, argn, gnB, Imn, hmnB, binom
    Qn = Q0 + n * Qg
    sn2 = s0 * s0 + n * sg2
    sn = sqrt(sn2)
    argn = (x - Qn) / sn
    gnB = 1.0 / (sqrt(2.0 * M_PI) * sn) * exp(-0.5 * argn * argn)
    SRn = pow(1.0 - n_omega, n) * gnB
    
    for m in range(1, n + 1):
        Qmn = Q0 + (n - m) * Qg
        smn2 = s0 * s0 + (n - m) * sg2
        smn = sqrt(smn2)
        
        cmn = n_alpha * pow(n_alpha * smn * sqrt(2.0), m - 1.0) / mt.factorial(m - 1)
        
        psi = (x - Qmn) / (sqrt(2.0) * smn)
        psi2 = psi * psi
        omega = (x - Qmn - n_alpha * smn * smn) / (sqrt(2.0) * smn)
        omega2 = omega * omega
        
        Imn = 0.0
        hi_limit = 25.0
        
        if omega >= hi_limit:
            Imn = exp(omega2 - psi2 + (m - 1.0) * log(omega))
        elif 0.0 <= omega < hi_limit:
            t1 = gamma(m / 2.0) * hyp1f1(1.0 / 2.0 - m / 2.0, 1.0 / 2.0, -omega2)
            t2 = 2.0 * omega * gamma((m + 1.0) / 2.0) * hyp1f1(3.0 / 2.0 - (m + 1.0) / 2.0, 3.0 / 2.0, -omega2)
            Imn = (1.0 / 2.0 / sqrt(M_PI)) * (t1 + t2) * exp(omega2 - psi2)
        elif omega < 0.0:
            t3 = 1.0 / (2.0 * M_PI) * gamma(m / 2.0) * gamma((m + 1.0) / 2.0)
            Imn = t3 * hyperu(m / 2.0, 1.0 / 2.0, omega2) * exp(-psi2)
        
        hmnB = cmn * Imn
        binom = mt.factorial(n) / mt.factorial(m) / mt.factorial(n - m)
        SRn += binom * pow(n_omega, m) * pow(1.0 - n_omega, n - m) * hmnB

    return SRn

cpdef double _fit_function_single_x(double x, double Q0, double s0, double Q1,
                                   double s1, double poisson_mean, double height, double n_omega,
                                   double n_alpha):

    cdef double total_poisson_calc =  _poisson(poisson_mean, 0)
    cdef double result = _SR0(x, Q0, s0, poisson_mean) * total_poisson_calc
    
    

    cdef double poisson_1 = _poisson(poisson_mean, 1)
    total_poisson_calc += poisson_1
    cdef double SR1 = _SR1(x,  Q0, s0, Q1, s1, poisson_mean, height, n_omega, n_alpha) * poisson_1

    result += SR1

    cdef double SRn
    for n in range(2, 10):
        poisson_n = _poisson(poisson_mean, n)
        SRn = _SRn(x,  Q0, s0, Q1, s1, poisson_mean, height, n_omega, n_alpha, n) * poisson_n
        total_poisson_calc += poisson_n
        if total_poisson_calc > 0.9999: #Break sum when 99.99% of poisson distribution is accounted for
            break
        result += SRn  # n >= nlim
    
    return result*height


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
    cdef double x1 = x[0]
    cdef double x2 = x[1]
    cdef double y1 = 0
    cdef double y2 = 0
    cdef double poisson                             
    if N_start == 0:
        poisson =  _poisson(poisson_mean, 0)
        y1 += _SR0(x1, Q0, s0, poisson_mean) * poisson
        y2 += _SR0(x2, Q0, s0, poisson_mean) * poisson
    
    if N_start <= 1:
        poisson = _poisson(poisson_mean, 1)
        y1 += _SR1(x1,  Q0, s0, Q1, s1, poisson_mean, height, n_omega, n_alpha) * poisson
        y2 += _SR1(x2,  Q0, s0, Q1, s1, poisson_mean, height, n_omega, n_alpha) * poisson


    for n in range(N_start, N_stop):
        if n in [0,1]:
            continue
        poisson = _poisson(poisson_mean, n)
        y1 += _SRn(x1,  Q0, s0, Q1, s1, poisson_mean, height, n_omega, n_alpha, n) * poisson
        y2 += _SRn(x2,  Q0, s0, Q1, s1, poisson_mean, height, n_omega, n_alpha, n) * poisson

    return (y1+y2)*height*(x2-x1)/2.


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
     
