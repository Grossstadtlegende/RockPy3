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
        for_substraction = [header.index(i) for i in ['x', 'y', 'z', 'M']]

        af3 = np.array([i for i in data if np.isnan(i[header.index('par2')])])
        data = np.array([i for i in data if not np.isnan(i[header.index('par2')])])

        # GENERATE RockPy Data object for data
        data = RockPy3.Data(data=data, column_names=list(map(str.lower, header)))
        data.rename_column('m', 'mag')
        data.rename_column('meas. time', 'time')
        data.define_alias('m', ('x', 'y', 'z'))

        if af3.any():
            af3 = RockPy3.Data(data=af3, column_names=list(map(str.lower, header)))
            af3.rename_column('m', 'mag')
            af3.rename_column('meas. time', 'time')
            af3.define_alias('m', ('x', 'y', 'z'))
        else:
            af3 = None

        out = {'data': data,
               'af3': af3}
        return out

    @correction
    def correct_af3(self, recalc_mag=True):
        """
        Subtracts the af3 data from the acquitition data
        """
        if self.data['af3']:
            self.log.info('FOUND AF3 measurement, subtracting AF3')
            self.data['data']['m'] = self.data['data']['m'].v - self.data['af3']['m'].v

            if recalc_mag:
                self.data['data']['mag'] = self.data['data'].magnitude(key='m')
            else:
                self.data['data']['mag'] = self.data['data']['mag'].v - self.data['af3']['mag'].v

    @property
    def cumulative(self):
        out = deepcopy(self.data['data'])
        out['m'] = np.cumsum(out['m'].v, axis=0)
        out['mag'] = out.magnitude(key='m')
        return out


class Arm_Acquisition(Acquisition):
    def __init__(self, sobj,
                 fpath=None, ftype=None,
                 mdata=None,
                 series=None,
                 idx=None,
                 initial_state=None,
                 ismean=False, base_measurements=None,
                 color=None, marker=None, linestyle=None,
                 ):
        super(Arm_Acquisition, self).__init__(sobj=sobj,
                                              fpath=fpath, ftype=ftype,
                                              mdata=mdata,
                                              series=series,
                                              idx=idx,
                                              initial_state=initial_state,
                                              ismean=ismean, base_measurements=base_measurements,
                                              color=color, marker=marker, linestyle=linestyle,
                                              )
        self.correct_af3()

    def format_sushibar(ftype_data, sobj_name=None):
        out = super(Arm_Acquisition, Arm_Acquisition).format_sushibar(ftype_data=ftype_data, sobj_name=sobj_name)
        for dtype in out:
            out[dtype].rename_column('par1', 'max_field')
            out[dtype].rename_column('par2', 'dc_field')
            out[dtype].rename_column('par3', 'window_upper')
            out[dtype].rename_column('par4', 'window_lower')
            out[dtype].rename_column('par5', 'dir')
            mean = np.mean(np.c_[out[dtype]['window_upper'].v, out[dtype]['window_lower'].v], axis=1)
            out[dtype] = out[dtype].append_columns(column_names='window_mean', data=mean)
            out[dtype].define_alias('variable', 'window_mean')
        return out
