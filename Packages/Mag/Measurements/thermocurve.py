__author__ = 'volk'
import RockPy3

from copy import deepcopy
from math import tanh, cosh

import numpy as np
import numpy.random
import scipy as sp
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from lmfit import minimize, Parameters, report_fit
import matplotlib.dates
import datetime

from RockPy3.core import measurement
from RockPy3.core.measurement import calculate, result, correction
from RockPy3.core.data import RockPyData
from pprint import pprint


class Thermocurve(measurement.Measurement):
    @staticmethod
    def format_mpms(ftype_data, sobj_name=None):
        if ftype_data.name == sobj_name:
            temp_index = ftype_data.units.index('K')

            # see if it a cooling or a warming curve
            mean_temp_difference_between_steps = np.mean(np.diff(ftype_data.data[:, temp_index]))
            if mean_temp_difference_between_steps > 0:
                dtype = 'warming'
            else:
                dtype = 'cooling'

            mdata = RockPyData(column_names=ftype_data.header,
                           data=ftype_data.data,
                           units=ftype_data.units
                           )
            mdata.rename_column('Temperature', 'temp')
            mdata.rename_column('Long Moment', 'mag')
            data = {dtype: mdata}
            return data
        else:
            RockPy3.logger.error('SAMPLE name of file does not match specified sample_name. << {} != {} >>'.format(ftype_data.name, sobj_name))

    @staticmethod
    def format_vftb(ftype_data, sobj_name=None):
        data = ftype_data.data
        units = ftype_data.units
        header = ftype_data.header
        mdata = {'warming':None, 'cooling':None}

        if len(data) > 2:
            RockPy3.logger.warning('LENGTH of machine.out_thermocurve =! 2. Assuming data[0] = heating data[1] = cooling')
        if 3 > len(data) > 1:
            for idx, dtype in enumerate(('warming', 'cooling')):
                mdata[dtype] = RockPyData(column_names=header, data=data[idx], units=units)
        else:
            RockPy3.logger.error('LENGTH of machine.out_thermocurve < 2.')
        return mdata

    def get_tc(self):
        for idx, dtype in enumerate(('warming', 'cooling')):
            self.data[dtype].find_peaks()
if __name__ == '__main__':
    file = '/Users/Mike/Dropbox/XXXOLD_BACKUPS/__PHD/__Projects/002 Hematite Nanoparticles, Morin Transition/04 data/MPMS/S3M2/S3M2_IRM7T_0T_60_300K_Cooling.rso.dat'
    file = '/Users/Mike/Dropbox/experimental_data/pyrrhotite/Pyrr17591-a_Ms_1T_300K_10K_Cooling.rso.dat'
    vftb_rmp = '/Users/Mike/Dropbox/experimental_data/pyrrhotite/VFTB/msm17591final.rmp'
    RockPy3.logger.setLevel('ERROR')
    S = RockPy3.Study

    #vftb sample
    s = S.add_sample(name='MSM17591')
    s.add_measurement(mtype='thermocurve', fpath=vftb_rmp, ftype='vftb')

    # mpms sample
    # s = S.add_sample(name='pyrr17591_a')
    # s.add_measurement(mtype='thermocurve', fpath=file, ftype='mpms')
    #
    fig = RockPy3.Figure()
    v = fig.add_visual(visual='thermocurve', visual_input=s)
    fig.show()