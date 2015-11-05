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

from RockPy3.core import measurement
import RockPy3.Packages.Mag.Measurements.demagnetization
import RockPy3.Packages.Mag.Measurements.acquisition
from RockPy3.core.measurement import calculate, result, correction
from RockPy3.core.data import RockPyData
from pprint import pprint


class Paleointensity(measurement.Measurement):

    @classmethod
    def from_measurement(cls, sobj, mobj,
                         initial_state = None, series=None, ismean=False,
                         color = None, marker=None, linestyle=None):
        """
        Takes a tuple of measurements and combines the data into a mdata dictionary
        """
        acquisition_data, demagnetization_data = None, None
        print(mobj)
        for m in mobj:
            if isinstance(m, RockPy3.Packages.Mag.Measurements.demagnetization.Demagnetization):
                demagnetization_data = m.data['data']
            if isinstance(m, RockPy3.Packages.Mag.Measurements.acquisition.Acquisition):
                if type(m) == RockPy3.Packages.Mag.Measurements.acquisition.Parm_Acquisition:
                    acquisition_data = m.cumulative
                else:
                    acquisition_data = m.data['data']

        if not any([acquisition_data, demagnetization_data]):
            cls.log.error(
                'ONE of the measurements does not match. A demagnetization and a acquisition object is needed')
            return

        mdata = {'acquisition': acquisition_data,
                 'demagnetization': demagnetization_data}
        return cls(sobj=sobj, ftype='from_measurement', mdata=mdata,
                   initial_state=initial_state, series=series, ismean=ismean,
                   color=color, marker=marker, linestyle=linestyle)

    ####################################################################################################################
    """ RESULTS CALCULATED USING CALCULATE_SLOPE  METHODS """

    @calculate
    def calculate_slope(self, var_min=20, var_max=90, component='mag',
                        **non_calculation_parameters):
        """
        calculates the least squares slope for the specified temperature interval

        :param parameter:

        """

        # get equal temperature steps for both th and ptrm measurements
        equal_steps = list(set(self.data['demagnetization']['variable'].v) & set(self.data['acquisition']['variable'].v))

        # Filter data for the equal steps and filter steps outside of tmin-tmax range
        # True if step between t_min, t_max
        demag_steps = (var_min <= self.data['demagnetization']['variable'].v) & (self.data['demagnetization']['variable'].v <= var_max)
        acq_steps = (var_min <= self.data['acquisition']['variable'].v) & (self.data['acquisition']['variable'].v <= var_max)

        demag_data = self.data['demagnetization'].filter(demag_steps)  # filtered data for t_min t_max
        acq_data = self.data['acquisition'].filter(acq_steps)  # filtered data for t_min t_max

        # filtering for equal variables
        th_idx = [i for i, v in enumerate(demag_data['variable'].v) if v in equal_steps]
        ptrm_idx = [i for i, v in enumerate(acq_data['variable'].v) if v in equal_steps]

        demag_data = demag_data.filter_idx(th_idx)  # filtered data for equal t(th) & t(ptrm)
        acq_data = acq_data.filter_idx(ptrm_idx)  # filtered data for equal t(th) & t(ptrm)

        data = RockPyData(['demagnetization', 'acquisition'])

        # setting the data
        data['demagnetization'] = demag_data[component].v
        data['acquisition'] = acq_data[component].v

        slope, sigma, y_int, x_int = data.lin_regress('acquisition', 'demagnetization')

        self.results['slope'] = [[[slope, sigma]]]
        self.results['sigma'] = sigma
        self.results['y_int'] = y_int
        self.results['x_int'] = x_int
        self.results['n'] = len(demag_data[component].v)

    @result
    def result_slope(self, t_min=20, t_max=700, component='mag',
                     recalc=False,
                     **non_calculation_parameters):
        """
        Gives result for calculate_slope(t_min, t_max), returns slope value if not calculated already
        """
        pass

    @result
    def result_n(self, t_min=20, t_max=700, component='mag',
                 recalc=False,
                 calculation_method='slope',
                 **non_calculation_parameters):
        """
        Number of steps used for the calculation of the best fit line
        """
        pass

    @result
    def result_sigma(self, t_min=20, t_max=700, component='mag',
                     recalc=False,
                     calculation_method='slope',
                     **non_calculation_parameters):
        """
        Standard deviation of the best fit line
        """
        pass

    @result
    def result_x_int(self, t_min=20, t_max=700, component='mag',
                     recalc=False,
                     calculation_method='slope',
                     **non_calculation_parameters):
        """
        """  # todo write doc
        pass

    @result
    def result_y_int(self, t_min=20, t_max=700, component='mag',
                     recalc=False,
                     calculation_method='slope',
                     **non_calculation_parameters):
        """
        """  # todo write doc
        pass


if __name__ == '__main__':
    step1C = '/Users/mike/Dropbox/experimental_data/RelPint/Step1C/1c.csv'
    step1B = '/Users/mike/Dropbox/experimental_data/RelPint/Step1B/IG_1291A.cmag.xml'

    S = RockPy3.Study
    s = S.add_sample(name='IG_1291A')
    pARM_acq = s.add_measurement(mtype='parmacq', fpath=step1C, ftype='sushibar', series=[('ARM', 50, 'muT')])
    NRM_AF = s.add_measurement(mtype='afdemag', fpath=step1B, ftype='cryo', series=[('NRM', 0, '')])

    m = s.add_measurement(mtype='paleointensity', mobj=(pARM_acq, NRM_AF))
    m.calc_all(var_min=30, var_max=60)
    print(m.results)

