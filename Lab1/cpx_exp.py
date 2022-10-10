# ESE 224 --- Signal and Information Processing
#
# Spring 2021
#
# Lab 1: Discrete sines, cosides and complex exponentials
#
# Question 1.1: Defining the complex exponential class
#
# By Vinicius Lima (adapted from 2020 edition of the course)

import numpy as np
import math
import cmath

class ComplexExp(object):
    """
    Creates a discrete complex exponential of discrete frequency k and duration N.
    Arguments:
        k: discrete frequency
        N: duration of the complex exponential 
    """

    def __init__(self, k, N):
        assert N > 0, "N should be a nonnegative scalar"
        self.k = k
        self.N = N

        # Vector containing elements of time indexes
        self.n = np.arange(N)

        # Vector containing elements of the complex exponential
        self.exp_kN = np.exp(2j*cmath.pi*self.k*self.n / self.N)
        self.exp_kN *= 1 / (np.sqrt(N))

        # Vector containing real elements of the complex exponential
        self.exp_kN_real = self.exp_kN.real

        # Vector containing imaginary elements of the complex exponential
        self.exp_kN_imag = self.exp_kN.imag


