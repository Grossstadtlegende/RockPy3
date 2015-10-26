__author__ = 'volk'
from copy import deepcopy
from math import tanh, cosh

import numpy as np
import numpy.random
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from lmfit import minimize, Parameters, report_fit

import RockPy3
from RockPy3.core import measurement
from RockPy3.core.measurement import calculate, result, correction
from RockPy3.core.data import RockPyData
from pprint import pprint

class Demagnetization(measurement.Measurement):

    @staticmethod
    def format_cryomag(ftype_data):
        pprint(ftype_data.raw_data)

class AfDemagnetization(Demagnetization):
    pass

class ThermalDemagnetization(Demagnetization):
    pass