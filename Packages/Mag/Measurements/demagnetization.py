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
import datetime

import RockPy3
from RockPy3.core import measurement
from RockPy3.core.measurement import calculate, result, correction
from RockPy3.core.data import RockPyData
from pprint import pprint


class Demagnetization(measurement.Measurement):

    @staticmethod
    def format_cryomag(ftype_data, sobj_name=None):
        if not sobj_name in ftype_data.raw_data:
            RockPy3.log.error('CANT find sample name << {} >> in file'.format(sobj_name))
            return None
        raw_data = ftype_data.raw_data[sobj_name]['stepdata']

        header = ['step', 'D', 'I', 'M', 'X', 'Y', 'Z', 'a95', 'sM', 'time']
        data = []
        for d in raw_data:
            aux = [d['step']]
            aux_data = [d['results'][h] for h in header[1:]]
            aux.extend(aux_data)
            data.append(aux)
        data = np.array(data).astype(float)
        data = RockPy3.Data(data=data, column_names=list(map(str.lower, header)))
        data.rename_column('m', 'mag')
        data.define_alias('m', ('x', 'y', 'z'))
        out = {'data': data}
        return out


class AfDemagnetization(Demagnetization):
    def format_cryomag(ftype_data, sobj_name=None):
        out = super(AfDemagnetization, AfDemagnetization).format_cryomag(ftype_data, sobj_name=sobj_name)
        for dtype in out:
            out[dtype].rename_column('step', 'field')
        return out


class ThermalDemagnetization(Demagnetization):
    def format_cryomag(ftype_data, sobj_name=None):
        out = super(ThermalDemagnetization, ThermalDemagnetization).format_cryomag(ftype_data, sobj_name=sobj_name)
        for dtype in out:
            out[dtype].rename_column('step', 'temp')
        return out
