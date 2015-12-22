__author__ = 'volk'
from copy import deepcopy
from math import tanh, cosh

import numpy as np
import numpy.random
import scipy as sp
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
import matplotlib.pyplot as plt


class Demagnetization(measurement.Measurement):
    ####################################################################################################################
    # FORMATTING
    @staticmethod
    def format_sushibar(ftype_data, sobj_name=None):
        if not sobj_name in ftype_data.raw_data:
            return
        data = np.nan_to_num(ftype_data.raw_data[sobj_name])
        header = ftype_data.header

        # data = np.array([i for i in data if not np.isnan(i[header.index('par2')])])
        # print(data)
        # GENERATE RockPy Data object for data
        data = RockPy3.Data(data=data, column_names=list(map(str.lower, header)))
        data.rename_column('m', 'mag')
        data.rename_column('meas. time', 'time')
        data.define_alias('m', ('x', 'y', 'z'))
        data.define_alias('variable', 'par1')

        out = {'data': data,
               }
        return out

    @staticmethod
    def format_cryomag(ftype_data, sobj_name=None):
        if not sobj_name in ftype_data.raw_data:
            RockPy3.logger.warning('CANT find sample name << {} >> in file'.format(sobj_name))
            sobj_name = list(ftype_data.raw_data.keys())[0]
            # raise ValueError('no sample named << {} >> in file'.format(sobj_name))
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
        data.define_alias('variable', 'step')
        out = {'data': data}
        return out

    ####################################################################################################################
    """ M1/2 """

    @calculate
    def calculate_m05_NONLINEAR(self, no_points=4, component='mag', check=False, **non_method_parameters):
        """
        """
        d = self.data['data'][component].v
        # get maximal moment
        mx_ind = np.argmax(np.fabs(d))
        mx = d[mx_ind]

        dnorm = d / mx
        # get limits for a calculation using the no_points points closest to 0 fro each direction
        ind = np.argmin(np.fabs(dnorm - 0.5))

        if dnorm[ind] > 0.5:
            max_idx = ind + no_points / 2
            min_idx = max_idx - no_points
        else:
            min_idx = ind - no_points / 2
            max_idx = min_idx + no_points
        variables = self.data['data']['variable'].v[min_idx:max_idx]
        data_points = dnorm[min_idx:max_idx]

        # generate new x from variables
        x = np.linspace(min(variables), max(variables), 10000)
        spl = UnivariateSpline(variables, data_points)
        y_new = spl(x)
        idx = np.argmin(abs(y_new - 0.5))
        result = abs(x[idx])

        if check:
            plt.plot(self.data['data']['variable'].v, dnorm, '.')
            plt.plot(x, y_new, '--')
            plt.plot(result, 0.5, 'xk')
            plt.grid()
            plt.show()

        # set result so it can be accessed
        self.results['m05'] = [[(np.nanmean(result), np.nan)]]

    @result
    def result_m05(self, recipe='nonlinear', recalc=False, **non_method_parameters):
        """
        the variable, where the moment is 1/2 of the max moment
        """
        pass

    @correction
    def correct_last_step(self, recalc_mag=True):
        """
        Subtracts the x,y,z values of the last step from the rest of the data
        """
        last_step = self.get_last_step()
        self.data['data']['m'] = self.data['data']['m'].v - last_step['m'].v
        if recalc_mag:
            self.data['data']['mag'] = self.data['data'].magnitude('m')
        else:
            self.data['data']['mag'] = self.data['data']['mag'].v - last_step['mag'].v

    @correction
    def correct_arbitrary_data(self, xyz, mag=None, recalc_mag=True):
        self.data['data']['m'] = self.data['data']['m'].v - xyz
        if recalc_mag:
            self.data['data']['mag'] = self.data['data'].magnitude('m')
        else:
            self.data['data']['mag'] = self.data['data']['mag'].v - mag

    def get_last_step(self):
        nsteps = len(self.data['data']['variable'].v)
        # get the rPdata of the last step
        last_step = self.data['data'].filter_idx(nsteps - 1)
        return last_step


class AfDemagnetization(Demagnetization):
    def format_cryomag(ftype_data, sobj_name=None):
        out = super(AfDemagnetization, AfDemagnetization).format_cryomag(ftype_data, sobj_name=sobj_name)
        if not out:
            return
        for dtype in out:
            out[dtype].define_alias('field', 'step')
            out[dtype].define_alias('variable', 'step')
        return out


class ThermalDemagnetization(Demagnetization):
    def format_cryomag(ftype_data, sobj_name=None):
        out = super(ThermalDemagnetization, ThermalDemagnetization).format_cryomag(ftype_data, sobj_name=sobj_name)
        for dtype in out:
            out[dtype].define_alias('temp', 'step')
            out[dtype].define_alias('variable', 'step')
        return out
