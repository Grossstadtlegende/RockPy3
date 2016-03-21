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
from collections import OrderedDict

from RockPy3.core import measurement
from RockPy3.core.measurement import calculate, result, correction
from RockPy3.core.data import RockPyData


class Thermocurve(measurement.Measurement):
    _visuals = (('thermocurve',{}), )

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
        mdata = OrderedDict(warming=None, cooling=None)

        if len(data) > 2:
            RockPy3.logger.warning('LENGTH of machine.out_thermocurve =! 2. Assuming data[0] = heating data[1] = cooling')
        if 3 > len(data) > 1:
            for idx, dtype in enumerate(('warming', 'cooling')):
                mdata[dtype] = RockPyData(column_names=header, data=data[idx], units=units)
        else:
            RockPy3.logger.error('LENGTH of machine.out_thermocurve < 2.')
        return mdata

    @staticmethod
    def format_vsm(ftype_data, sobj_name=None):

        header = ftype_data.header
        segments = ftype_data.get_segments_from_data()
        data = ftype_data.get_data()

        mdata = OrderedDict()

        aux = np.array([j for i in data for j in i])  # combine all data arrays
        a = np.array([(i, v) for i, v in enumerate(np.diff(aux, axis=0)[:, 0])])

        sign = np.sign(np.diff(aux, axis=0)[:, 1])

        threshold = 3
        zero_crossings = [i + 1 for i in range(len(a[:, 1]) - 1)
                          if a[i, 1] > 0 > a[i + 1, 1] and a[i, 1] > 0 > a[i + 2, 1]
                          or a[i, 1] < 0 < a[i + 1, 1] and a[i, 1] < 0 < a[i + 2, 1]]
        zero_crossings = [0] + zero_crossings  # start with zero index
        zero_crossings += [len(aux)]  # append last index

        ut = 0  # running number warming
        dt = 0  # running number cooling

        for i, v in enumerate(zero_crossings):
            if v < zero_crossings[-1]:  # prevents index Error
                if sum(a[v:zero_crossings[i + 1], 1]) < 0:  # cooling
                    name = 'cooling%02i' % (ut)
                    ut += 1
                else:
                    name = 'warming%02i' % (dt)
                    dt += 1
                data = aux[v:zero_crossings[i + 1] + 1]
                rpd = RockPyData(column_names=header, data=data)
                rpd.rename_column('temperature', 'temp')
                rpd.rename_column('moment', 'mag')
                mdata.update({name: rpd})

        return mdata
    # def get_tc(self):
    #     for idx, dtype in enumerate(('warming', 'cooling')):
    #         self.data[dtype].find_peaks()

    def combine_measurements(self, others, remove_others = False, normalize_to_last=False):
        """
        it combines several measurements into the specified one. e.g. a second cooling run at the end of the measurement
        Parameters
        ----------

        others: list
            the measurements to appended to the self.data dictionary. The names are chosen according to the total number of heating and cooling runs

        remove_others: bool
            default: False
            if True the measurements are removed from the sample after it is combined with the measurement
            if False the measurements only combined

        normalize_to_last:
            default: False
            nomalizes the data to be combined to the last point of the last segment. So that different calibrations do not affect the measurement

        Returns
        -------

        """
        others = RockPy3._to_tuple(others)
        self.log.info('COMBINING << {} >> with {}'.format(self, others))
        cool_idx = sum(1 for i in self.data if 'cool' in i)
        warm_idx = sum(1 for i in self.data if 'warm' in i)
        c,w = cool_idx-1, warm_idx-1 # old maximum indices

        nfactor = None
        last_segment = list(self.data.keys())[-1]

        if normalize_to_last:
            nfactor = self.data[last_segment]['mag'].v[-1]

        for m in others:
            m = deepcopy(m)
            if normalize_to_last:
                m.normalize('cooling00')
                m.normalize(norm_factor=1/nfactor)

            for dtype in m.data:
                if 'cool' in dtype:
                    self.data.update({'cooling%02i'%cool_idx: m.data[dtype]})
                    cool_idx += 1
                if 'warming' in dtype:
                    self.data.update({'warming%02i'%warm_idx: m.data[dtype]})
                    warm_idx += 1

            if remove_others:
                self.log.info('REMOVING << {} >> from sample << {} >>'.format(self, self.sobj))
                self.sobj.remove_measurement(mobj=m)

        return self

if __name__ == '__main__':

    S = RockPy3.Study

    """ vftb sample """
    # s = S.add_sample(name='MSM17591')
    # s.add_measurement(mtype='thermocurve', fpath=vftb_rmp, ftype='vftb')

    """ mpms sample """
    # s = S.add_sample(name='pyrr17591_a')
    # s.add_measurement(mtype='thermocurve', fpath=file, ftype='mpms')

    """ vsm sample """
    s = S.add_sample('167a', sgroup='LTPY')
    s.add_measurement(fpath='/Users/mike/Dropbox/experimental_data/RMP/LTPY/LTPY_167a_RMP_VSM#[]_[]_[]##STD001.000')


    fig = RockPy3.Figure()
    v = fig.add_visual(visual='thermocurve', data=s)
    fig.show()