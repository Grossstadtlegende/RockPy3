__author__ = 'volk'
import RockPy3

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


from RockPy3.core.measurement import calculate, result, correction
from RockPy3.core.data import RockPyData
from pprint import pprint


class Acquisition(RockPy3.core.measurement.Measurement):
    @staticmethod
    def format_sushibar(ftype_data, sobj_name=None):
        if not sobj_name in ftype_data.raw_data:
            return

        data = ftype_data.raw_data[sobj_name]
        # data = np.nan_to_num(ftype_data.raw_data[sobj_name])
        header = ftype_data.header

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
            # print('Moment before correction')
            # print(self.data['data']['m'])
            # print('Moment of the AF3')
            # print(self.data['af3']['m'])
            self.data['data']['m'] = self.data['data']['m'].v - self.data['af3']['m'].v
            # print('Moment after correction')
            # print(self.data['data']['m'])

            if recalc_mag:
                self.data['data']['mag'] = self.data['data'].magnitude(key='m')
            else:
                self.data['data']['mag'] = self.data['data']['mag'].v - self.data['af3']['mag'].v
    @correction
    def correct_arbitrary_data(self, xyz, mag=None, recalc_mag=True):
        self.data['data']['m'] = self.data['data']['m'].v - xyz
        if recalc_mag:
            self.data['data']['mag'] = self.data['data'].magnitude('m')
        else:
            self.data['data']['mag'] = self.data['data']['mag'].v - mag

    @property
    def cumulative(self):
        out = deepcopy(self.data['data'])
        out = out.append_rows(data=[0 for i in out.column_names]).sort('variable')
        out['m'] = np.cumsum(out['m'].v, axis=0)
        out['mag'] = out.magnitude(key='m')
        out.define_alias('variable', 'window_upper')
        return out

class Parm_Acquisition(Acquisition):
    def __init__(self, sobj,
                 fpath=None, ftype=None,
                 mdata=None,
                 series=None,
                 idx=None,
                 initial_state=None,
                 ismean=False, base_measurements=None,
                 color=None, marker=None, linestyle=None,
                 ):
        super(Parm_Acquisition, self).__init__(sobj=sobj,
                                              fpath=fpath, ftype=ftype,
                                              mdata=mdata,
                                              series=series,
                                              idx=idx,
                                              initial_state=initial_state,
                                              ismean=ismean, base_measurements=base_measurements,
                                              color=color, marker=marker, linestyle=linestyle,
                                              )
        # self.correct_af3()

    def format_sushibar(ftype_data, sobj_name=None):
        out = super(Parm_Acquisition, Parm_Acquisition).format_sushibar(ftype_data=ftype_data, sobj_name=sobj_name)
        if not out:
            return
        for dtype in out:
            if out[dtype] is not None:
                out[dtype].rename_column('par1', 'max_field')
                out[dtype].rename_column('par2', 'dc_field')
                out[dtype].rename_column('par3', 'window_upper')
                out[dtype].rename_column('par4', 'window_lower')
                out[dtype].rename_column('par5', 'dir')
                mean = np.mean(np.c_[out[dtype]['window_upper'].v, out[dtype]['window_lower'].v], axis=1)
                out[dtype] = out[dtype].append_columns(column_names='window_mean', data=mean)
                out[dtype].define_alias('variable', 'window_mean')
        return out

if __name__ == '__main__':
    step1C = '/Users/mike/Dropbox/experimental_data/RelPint/Step1C/1c.csv'
    S = RockPy3.Study
    s = S.add_sample(name='IG_1291A')
    ARM_acq = s.add_measurement(mtype='armacq', fpath=step1C, ftype='sushibar', series=[('ARM', 50, 'muT')])
