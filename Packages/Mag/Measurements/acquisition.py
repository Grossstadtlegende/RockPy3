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
import matplotlib.dates
import datetime

import RockPy3
from RockPy3.core import measurement
from RockPy3.core.measurement import calculate, result, correction
from RockPy3.core.data import RockPyData
from pprint import pprint


class Acquisition(measurement.Measurement):

    @staticmethod
    def format_sushibar(ftype_data, sobj_name=None):
        if not sobj_name in ftype_data.raw_data:
            RockPy3.log.error('CANT find sample name << {} >> in file'.format(sobj_name))
            return None
        data = ftype_data.raw_data[sobj_name]
        header = ftype_data.header

        af3 = [i for i in data if np.isnan(i[header.index('par2')])]
        data = np.array([i for i in data if not np.isnan(i[header.index('par2')])])

        if af3:
            RockPy3.logger.info('FOUND AF3 measurement, subtracting AF3')
            data = data - af3

        af3 = RockPy3.Data(data=af3, column_names=list(map(str.lower, header)))
        af3.rename_column('m', 'mag')
        af3.define_alias('m', ('x', 'y', 'z'))

        data = RockPy3.Data(data=data, column_names=list(map(str.lower, header)))
        data.rename_column('m', 'mag')
        data.define_alias('m', ('x', 'y', 'z'))


        out = {'data': data,
               'af3': af3}
        return out

class Arm_Acquisition(Acquisition):

    def format_sushibar(ftype_data, sobj_name=None):
        out = super(Arm_Acquisition, Arm_Acquisition).format_sushibar(ftype_data=ftype_data, sobj_name=sobj_name)
        for dtype in out:
            out[dtype].rename_column('par1', 'max_field')
            out[dtype].rename_column('par2', 'dc_field')
            out[dtype].rename_column('par3', 'window_upper')
            out[dtype].rename_column('par4', 'window_lower')
            out[dtype].rename_column('par5', 'dir')
            mean = np.mean(np.c_[out[dtype]['window_upper'].v, out[dtype]['window_upper'].v], axis=1)
            out[dtype] = out[dtype].append_columns(column_names='window_mean', data=mean)
        return out

