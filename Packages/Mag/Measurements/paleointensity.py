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


class Paleointensity(measurement.Measurement):

    @classmethod
    def from_measurement(cls, sobj, mobj,
                         initial_state = None, series=None, ismean=False,
                         color = None, marker=None, linestyle=None):
        """
        Takes a tuple of measurements and combines the data into a mdata dictionary
        """
        acquisition_data, demagnetization_data = None, None
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
    def calculate_slope(self, var_min=20, var_max=700, component='mag',
                        **non_method_parameters):
        """
        calculates the least squares slope for the specified temperature interval

        :param parameter:

        """

        # get equal temperature steps for both demagnetization and acquisition measurements
        equal_steps = list(set(self.data['demagnetization']['variable'].v) & set(self.data['acquisition']['variable'].v))

        # Filter data for the equal steps and filter steps outside of tmin-tmax range
        # True if step between var_min, var_max
        demag_steps = (var_min <= self.data['demagnetization']['variable'].v) & (self.data['demagnetization']['variable'].v <= var_max)
        acq_steps = (var_min <= self.data['acquisition']['variable'].v) & (self.data['acquisition']['variable'].v <= var_max)

        demag_data = self.data['demagnetization'].filter(demag_steps)  # filtered data for var_min var_max
        acq_data = self.data['acquisition'].filter(acq_steps)  # filtered data for var_min var_max

        # filtering for equal variables
        demagnetization_idx = [i for i, v in enumerate(demag_data['variable'].v) if v in equal_steps]
        acquisition_idx = [i for i, v in enumerate(acq_data['variable'].v) if v in equal_steps]

        if not any([demagnetization_idx, acquisition_idx]):
            self.log.error('NOT enough data')
            return
        demag_data = demag_data.filter_idx(demagnetization_idx)  # filtered data for equal t(demagnetization) & t(acquisition)
        acq_data = acq_data.filter_idx(acquisition_idx)  # filtered data for equal t(demagnetization) & t(acquisition)

        data = RockPyData(['demagnetization', 'acquisition'])

        # setting the data
        data['demagnetization'] = demag_data[component].v
        data['acquisition'] = acq_data[component].v

        try:
            slope, sigma, y_int, x_int = data.lin_regress('acquisition', 'demagnetization')
            self.results['slope'] = [[[slope, sigma]]]
            self.results['sigma'] = sigma
            self.results['y_int'] = y_int
            self.results['x_int'] = x_int
            self.results['n'] = len(demag_data[component].v)
        except TypeError:
            self.log.error('No data found')



    @result
    def result_slope(self, var_min=20, var_max=700, component='mag', **non_method_parameters):
        """
        Gives result for calculate_slope(var_min, var_max), returns slope value if not calculated already
        """
        pass

    @result
    def result_n(self, var_min=20, var_max=700, component='mag', dependent='slope',
                 **non_method_parameters):
        """
        Number of steps used for the calculation of the best fit line
        """
        pass

    @result
    def result_sigma(self, var_min=20, var_max=700, component='mag', dependent='slope',
                     **non_method_parameters):
        """
        Standard deviation of the best fit line
        """
        pass

    @result
    def result_x_int(self, var_min=20, var_max=700, component='mag', dependent='slope',
                     **non_method_parameters):
        """
        """  # todo write doc
        pass

    @result
    def result_y_int(self, var_min=20, var_max=700, component='mag', dependent='slope',
                     **non_method_parameters):
        """
        """  # todo write doc
        pass

    ####################################################################################################################
    ''' NOT ONLY SLOPE BASED '''

    @calculate
    def calculate_b_anc(self, var_min=20, var_max=700, component='mag', b_lab=35.0,
                        **non_method_parameters):
        """
        calculates the :math:`B_{anc}` value for a given lab field in the specified temperature interval.

        :param parameter:

        Note
        ----
            This calculation method calls calculate_slope if you call it again afterwards, with different
            calculation_parameters, it will not influence this result. Therfore you have to be careful when calling
            this.
        """
        slope = self.result_slope(var_min=var_min, var_max=var_max, component=component, **non_method_parameters)
        self.results['b_anc'] = [[[abs(b_lab * slope[0]), abs(b_lab * slope[1])]]]
        self.results['sigma_b_anc'] = abs(b_lab * slope[1])

    @result
    def result_b_anc(self, var_min=20, var_max=700, component='mag', b_lab=35.0,
                     **non_method_parameters):  # todo write comment
        pass

    @result
    def result_sigma_b_anc(self, var_min=20, var_max=700, component='mag', b_lab=35.0,
                           dependent='b_anc',
                           **non_method_parameters):  # todo write comment
        pass

    ####################################################################################################################
    """ F """

    @calculate
    def calculate_f(self, var_min=20, var_max=700, **non_method_parameters):
        """

        The remanence fraction, f, was defined by Coe et al. (1978) as:

        .. math::

           f =  \\frac{\\Delta y^T}{y_0}

        where :math:`\Delta y^T` is the length of the NRM/TRM segment used in the slope calculation.


        :param parameter:
        :return:

        """
        delta_y_dash = self.delta_y_dash(**non_method_parameters)
        y_int = self.results['y_int'].v
        self.results['f'] = delta_y_dash / abs(y_int)

    @result
    def result_f(self, var_min=20, var_max=700, **options):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ F_VDS """

    @calculate
    def calculate_f_vds(self, var_min=20, var_max=700,
                        **non_method_parameters):
        """

        NRM fraction used for the best-fit on an Arai diagram calculated as a vector difference sum (Tauxe and Staudigel, 2004).

        .. math::

           f_{VDS}=\\frac{\Delta{y'}}{VDS}

        :param parameter:
        :return:

        """
        delta_y = self.delta_y_dash(var_min=var_min, var_max=var_max, **non_method_parameters)
        VDS = self.calculate_vds()
        self.results['f_vds'] = delta_y / VDS

    @result
    def result_f_vds(self, var_min=20, var_max=700, **non_method_parameters):
        pass

    ####################################################################################################################
    """ FRAC """

    @calculate
    def calculate_frac(self, var_min=20, var_max=700, **non_method_parameters):
        """

        NRM fraction used for the best-fit on an Arai diagram determined entirely by vector difference sum
        calculation (Shaar and Tauxe, 2013).

        .. math::

            FRAC=\\frac{\sum\limits_{i=start}^{end-1}{ \left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right| }}{VDS}

        :param parameter:
        :return:

        """

        NRM_sum = np.sum(np.fabs(self.calculate_vd(var_min=var_min, var_max=var_max, **non_method_parameters)))
        VDS = self.calculate_vds()
        self.results['frac'] = NRM_sum / VDS

    @result
    def result_frac(self, var_min=20, var_max=700, **options):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ BETA """

    @calculate
    def calculate_beta(self, var_min=20, var_max=700, component='mag',
                       **non_method_parameters):
        """

        :math:`\beta` is a measure of the relative data scatter around the best-fit line and is the ratio of the
        standard error of the slope to the absolute value of the slope (Coe et al., 1978)

        .. math::

           \\beta = \\frac{\sigma_b}{|b|}


        :param parameters:
        :return:

        """

        slope = self.result_slope(var_min=var_min, var_max=var_max, component=component, **non_method_parameters)[0]
        sigma = self.result_sigma(var_min=var_min, var_max=var_max, component=component, **non_method_parameters)[0]
        self.results['beta'] = sigma / abs(slope)

    @result
    def result_beta(self, var_min=20, var_max=700, **non_method_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ G """

    @calculate
    def calculate_g(self, var_min=20, var_max=700, component='mag',
                    **non_method_parameters):
        """

        Gap factor: A measure of the gap between the points in the chosen segment of the Arai plot and the least-squares
        line. :math:`g` approaches :math:`(n-2)/(n-1)` (close to unity) as the points are evenly distributed.

        """
        y_dash = self.y_dash(var_min=var_min, var_max=var_max, component=component, **non_method_parameters)
        delta_y_dash = self.delta_y_dash(var_min=var_min, var_max=var_max, component=component, **non_method_parameters)
        y_dash_diff = [(y_dash[i + 1] - y_dash[i]) ** 2 for i in range(len(y_dash) - 1)]
        y_sum_dash_diff_sq = np.sum(y_dash_diff, axis=0)

        self.results['g'] = 1 - y_sum_dash_diff_sq / delta_y_dash ** 2

    @result
    def result_g(self, var_min=20, var_max=700, component='mag', **non_method_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ GAP MAX """

    @calculate
    def calculate_gap_max(self, var_min=20, var_max=700, **non_method_parameters):
        """
        The gap factor is a measure of the average Arai plot point spacing and may not represent extremes
        of spacing. To account for this Shaar and Tauxe (2013)) proposed :math:`GAP_{\text{MAX}}`, which is the maximum
        gap between two points determined by vector arithmetic.

        .. math::
           GAP_{\\text{MAX}}=\\frac{\\max{\{\\left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right|\}}_{i=start, \\ldots, end-1}}
           {\\sum\\limits_{i=start}^{end-1}{\\left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right|}}

        :return:

        """
        vd = self.calculate_vd(var_min=var_min, var_max=var_max)
        max_vd = np.max(vd)
        sum_vd = np.sum(vd)
        self.results['gap_max'] = max_vd / sum_vd

    @result
    def result_gap_max(self, var_min=20, var_max=700, **non_method_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ Q """

    @calculate
    def calculate_q(self, var_min=20, var_max=700, component='mag', **non_method_parameters):
        """

        The quality factor (:math:`q`) is a measure of the overall quality of the paleointensity estimate and combines
        the relative scatter of the best-fit line, the NRM fraction and the gap factor (Coe et al., 1978).

        .. math::
           q=\\frac{\\left|b\\right|fg}{\\sigma_b}=\\frac{fg}{\\beta}

        :param parameter:
        :return:

        """
        self.log.debug('CALCULATING\t quality parameter')

        beta = self.result_beta(var_min=var_min, var_max=var_max, component=component)[0]
        f = self.result_f(var_min=var_min, var_max=var_max, component=component)[0]
        gap = self.result_g(var_min=var_min, var_max=var_max, component=component)[0]
        self.results['q'] = (f * gap) / beta

    @result
    def result_q(self, var_min=20, var_max=700, component='mag', **non_method_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ W """

    @calculate
    def calculate_w(self, var_min=20, var_max=700, component='mag', **non_method_parameters):
        """
        Weighting factor of Prevot et al. (1985). It is calculated by

        .. math::

           w=\\frac{q}{\\sqrt{n-2}}

        Originally it is :math:`w=\\frac{fg}{s}`, where :math:`s^2` is given by

        .. math::

           s^2 = 2+\\frac{2\\sum\\limits_{i=start}^{end}{(x_i-\\bar{x})(y_i-\\bar{y})}}
              {\\left( \\sum\\limits_{i=start}^{end}{(x_i- \\bar{x})^{\\frac{1}{2}}}
              \\sum\\limits_{i=start}^{end}{(y_i-\\bar{y})^2} \\right)^2}

        It can be noted, however, that :math:`w` can be more readily calculated as:

        .. math::

           w=\\frac{q}{\\sqrt{n-2}}

        :param parameter:
        """
        q = self.result_q(var_min=var_min, var_max=var_max, component=component)[0]
        n = self.result_n(var_min=var_min, var_max=var_max, component=component)[0]
        self.results['w'] = q / np.sqrt((n - 2))

    @result
    def result_w(self, var_min=20, var_max=700, component='mag', **options):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ HELPER FUNCTIONS FOR CALCULATE METHODS """

    def filter_demagnetization_ptrm(self, var_min=20, var_max=700):
        """
        Filters the th and ptrm data so that the temperatures are within var_min, var_max and only temperatures in both
        th and ptrm are returned.
        """
        idx_demagnetization_ptrm = np.array([(v1, i1, i2)
                                for i1, v1 in enumerate(self.data['demagnetization']['variable'].v)
                                for i2, v2 in enumerate(self.data['acquisition']['variable'].v)
                                if var_min <= v1 <= var_max
                                if var_min <= v2 <= var_max
                                if v1 == v2])

        y = self.data['demagnetization'].filter_idx(idx_demagnetization_ptrm[:, 1])  # filtered data for var_min var_max
        x = self.data['acquisition'].filter_idx(idx_demagnetization_ptrm[:, 2])  # filtered data for var_min var_max
        return y, x

    def calculate_vd(self, var_min=20, var_max=700,
                     **non_method_parameters):  # todo move in rockpydata?
        """
        Vector differences

        :param parameter:
        :return:

        """
        idx = (self.data['demagnetization']['variable'].v <= var_max) & (var_min <= self.data['demagnetization']['variable'].v)
        data = self.data['demagnetization'].filter(idx)
        vd = np.array([np.linalg.norm(i) for i in np.diff(data['m'].v, axis=0)])
        return vd

    def calculate_vds(self, **non_method_parameters):  # todo move in rockpydata?
        """
        The vector difference sum of the entire NRM vector :math:`\\mathbf{NRM}`.

        .. math::

           VDS=\\left|\\mathbf{NRM}_{n_{max}}\\right|+\\sum\\limits_{i=1}^{n_{max}-1}{\\left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right|}

        where :math:`\\left|\\mathbf{NRM}_{i}\\right|` denotes the length of the NRM vector at the :math:`i^{demagnetization}` step.

        Parameters
        ----------
            var_min: float
            var_max: float
            recalc: bool
            non_method_parameters: dict
        """
        # tmax = max(self.data['demagnetization']['variable'].v)
        # NRM_var_max = self.data['demagnetization'].filter(self.data['demagnetization']['variable'].v == tmax)['mag'].v[0]
        NRM_var_max = np.linalg.norm(self.data['demagnetization']['m'].v[-1])
        NRM_sum = np.sum(np.abs(self.calculate_vd(var_min=0, var_max=700)))
        return abs(NRM_var_max) + NRM_sum

    def x_dash(self, var_min=20, var_max=700, component='mag',
               **non_method_parameters):
        """

        :math:`x_0 and :math:`y_0` the x and y points on the Arai plot projected on to the best-fit line. These are
        used to
        calculate the NRM fraction and the length of the best-fit line among other parameters. There are
        multiple ways of calculating :math:`x_0 and :math:`y_0`, below is one example.

        ..math:

          x_i' = \\frac{1}{2} \\left( x_i + \\frac{y_i - Y_{int}}{b}


        :param parameter:
        :return:

        """

        demagnetization, acquisition = self.filter_demagnetization_ptrm(var_min=var_min, var_max=var_max)
        x_dash = (demagnetization[component].v - self.result_y_int(var_min=var_min, var_max=var_max, component=component)[0])
        x_dash = x_dash / self.result_slope(var_min=var_min, var_max=var_max, component=component)[0]
        x_dash = acquisition[component].v + x_dash
        x_dash = x_dash / 2.

        return x_dash

    def y_dash(self, var_min=20, var_max=700, component='mag',
               **non_method_parameters):
        """

        :math:`x_0` and :math:`y_0` the x and y points on the Arai plot projected on to the best-fit line. These are
        used to
        calculate the NRM fraction and the length of the best-fit line among other parameters. There are
        multiple ways of calculating :math:`x_0` and :math:`y_0`, below is one example.

        ..math:

           y_i' = \\frac{1}{2} \\left( x_i + \\frac{y_i - Y_{int}}{b}


        :param parameter:
        :return:

        """
        demagnetization, acquisition = self.filter_demagnetization_ptrm(var_min=var_min, var_max=var_max)

        y_dash = demagnetization[component].v + \
                 (self.result_slope(var_min=var_min, var_max=var_max, component=component)[0] * acquisition[component].v +
                  self.result_y_int(var_min=var_min, var_max=var_max, component=component)[0])
        y_dash = y_dash / 2
        return y_dash

    def delta_x_dash(self, var_min=20, var_max=700, component='mag',
                     **non_method_parameters):
        """

        :math:`\Delta x_0` is the TRM length of the best-fit line on the Arai plot.

        """
        x_dash = self.x_dash(var_min=var_min, var_max=var_max, component=component, **non_method_parameters)
        out = abs(np.max(x_dash) - np.min(x_dash))
        return out

    def delta_y_dash(self, var_min=20, var_max=700, component='mag',
                     **non_method_parameters):
        """

        :math:`\Delta y_0`  is the NRM length of the best-fit line on the Arai plot.

        """
        y_dash = self.y_dash(var_min=var_min, var_max=var_max, component=component, **non_method_parameters)
        out = abs(np.max(y_dash) - np.min(y_dash))
        return out

    def best_fit_line_length(self, var_min=20, var_max=700, component='mag'):
        L = np.sqrt((self.delta_x_dash(var_min=var_min, var_max=var_max, component=component)) ** 2 +
                    (self.delta_y_dash(var_min=var_min, var_max=var_max, component=component)) ** 2)
        return L

    ####################################################################################################################
if __name__ == '__main__':
    step1C = '/Users/mike/Dropbox/experimental_data/RelPint/Step_1C'
    step1B = '/Users/mike/Dropbox/experimental_data/RelPint/Step_1B'

    S = RockPy3.Study
    s = S.add_sample(name='IG_1291A')
    pARM_acq = s.add_measurement(mtype='parmacq', fpath=step1C, ftype='sushibar', series=[('ARM', 50, 'muT')])
    NRM_AF = s.add_measurement(mtype='afdemag', fpath=step1B, ftype='sushibar', series=[('NRM', 0, '')])

    m = s.add_measurement(mtype='paleointensity', mobj=(pARM_acq, NRM_AF))
    # print(m.results)
    # print(m.res_signature()['b_anc'])
    # m.result_slope(var_max=90, var_min=10)
    # print(m.result_beta())
    # m.calc_all(var_max=90, var_min=10)
    # print(m.results)
    # fig = RockPy3.Figure(fig_input=S)
    # v = fig.add_visual('paleointensity')
    # fig.show()

