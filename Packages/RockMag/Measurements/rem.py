__author__ = 'mike'
from copy import deepcopy

import numpy as np

from RockPy.core import measurement
from RockPy.core.measurement import calculate, result


class Rem_Prime(measurement.Measurement):
    def __init__(self, sample_obj,
                 ftype='combined', fpath=None, mtype='rem_prime',
                 af1=None, af2=None,
                 **options):
        """
        :param sample_obj:
        :param mtype:
        :param mfile:
        :param machine:
        :param mdata: when mdata is set, this will be directly used as measurement data without formatting from file
        :param options:
        :return:
        """
        self.af1 = af1
        self.af2 = af2

        data = {'af1': af1.data, 'af2': af2.data}
        super(Rem_Prime, self).__init__(sample_obj=sample_obj, mtype=mtype, fpath=fpath, ftype=ftype, mdata=data)

    ####################################################################################################################
    """ REM PRIME """
    @calculate
    def calculate_rem_prime(self, component='mag', b_min=0, b_max=90, interpolate=True, smoothing=0, **non_calculation_parameter):
        """
        :param parameter:
        """
        ratios = self.calc_ratios(b_min=b_min, b_max=b_max, interpolate=interpolate, smoothing=smoothing)
        self.results['rem_prime'] = np.mean(abs(ratios[component].v))

    @result
    def result_rem_prime(self):
        pass

    ####################################################################################################################
    """ REM PRIME """

    @calculate
    def calculate_rem(self, component='mag', **non_calculation_parameter):
        af1 = deepcopy(self.data['af1']['data'])
        af2 = deepcopy(self.data['af2']['data'])

        rem = af1[component].v[0] / af2[component].v[0]
        self.results['rem'] = rem

    @result
    def result_rem(self, recalc=False, **opt):
        pass

    ####################################################################################################################
    """ HELPER """
    def calc_ratios(self, b_min=0, b_max=90, component='mag', interpolate=True, smoothing=0):
        """
        calculates the ratio between the derivatives of two separate AF demagnetization measurements
        :param parameter:
        :return:
        """

        af1 = deepcopy(self.data['af1']['data'])
        af2 = deepcopy(self.data['af2']['data'])

        # truncate to within steps
        ### AF1
        idx = [i for i, v in enumerate(af1['field'].v) if b_min <= v <= b_max]
        if len(idx) != len(af1['field'].v):
            af1 = af1.filter_idx(idx)
        ### AF2
        idx = [i for i, v in enumerate(af2['field'].v) if b_min <= v <= b_max]
        if len(idx) != len(af2['field'].v):
            af2 = af2.filter_idx(idx)

        # find same fields
        b1 = set(af1['field'].v)
        b2 = set(af2['field'].v)

        if not b1 == b2:
            equal_fields = sorted(list(b1 & b1))
            interpolate_fields = sorted(list(b1 | b1))
            not_in_b1 = b1 - b2
            not_in_b2 = b2 - b1
            if interpolate:
                if not_in_b1:
                    af1 = af1.interpolate(new_variables=interpolate_fields)
                if not_in_b2:
                    af2 = af2.interpolate(new_variables=interpolate_fields)
            else:
                if not_in_b1:
                    idx = [i for i, v in enumerate(af1['field'].v) if v not in equal_fields]
                    af1 = af1.filter_idx(idx, invert=True)
                if not_in_b2:
                    idx = [i for i, v in enumerate(af2['field'].v) if v not in equal_fields]
                    af2 = af2.filter_idx(idx, invert=True)

        daf1 = af1.derivative(component, 'field', smoothing=smoothing)
        daf2 = af2.derivative(component, 'field', smoothing=smoothing)

        ratios = daf1 / daf2
        ratios.data = np.fabs(ratios.data) # no negatives
        return ratios

    def calc_rem_prime_field(self, component = 'mag', add2results = False, **parameter):
        ratios = self.calc_ratios(**parameter)
        for i, v in enumerate(ratios['field'].v):
            name = 'rem\' [%.1f]' % v
            self.results = self.results.append_columns(column_names=name, data=abs(ratios[component].v[i]))
        return ratios

