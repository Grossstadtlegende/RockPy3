# coding=utf-8
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

import RockPy3
from RockPy3.core import measurement
from RockPy3.core.measurement import calculate, result, correction
from RockPy3.core.data import RockPyData

import matplotlib.pyplot as plt

class Hysteresis(measurement.Measurement):
    """
    Measurement Class for Hysteresis Measurements

    **Corrections**

       correct_slope:
          *not implemented* corrects data for high-field susceptibility

       correct_hsym:
          *not implemented* corrects data for horizontal assymetry

       correct_vsym:
          *not implemented* corrects data for vertical assymetry

       correct_drift:
          *not implemented* corrects data for machine drift

       correct_pole_sat:
          *not implemented* corrects data for pole piece saturation

       correct_holder:
          *not implemented* corrects data for holder magnetization

       correct_outliers:
          *not implemented* removes outliers from data

    Results:

    Decomposition:

    **Fitting**

    See Also
    --------

       :cite:`Dobeneck1996a`
       :cite:`Fabian2003`
       :cite:`Yu2005b`
    """

    @classmethod
    def from_simulation(cls, sobj, idx=0,
                        ms=250., mrs_ms=0.5, bc=0.2, hf_sus=1., bmax=1.8, b_sat=1, steps=20,
                        noise=None, color=None, marker=None, linestyle=None):
        """
        Simulation of hysteresis loop using single tanh and sech functions.

        Parameters:
           m_idx (int): index of measurement
           ms (float): desired saturation magnetization
           mrs_ms (float): :math:`M_{rs}/M_{s}` ratio
           bc (float): desired bc
           hf_sus:

           bmax:

           b_sat: float
              Field at which 99% of the moment is saturated

           steps:

           sample:

           color:

           parameter:

           noise: float
            noise in percent

        :Returns:

        :Note:

        Increasing the Mrs/Ms ratio to more then 0.5 results in weird looking hysteresis loops

        :TODO:

           Not working properly, yet. Use with caution
        """

        cls.log.info('CREATING simulation measurement with {}'.format(locals()))

        data = {'up_field': None,
                'down_field': None,
                'virgin': None}

        fields = cls.get_grid(bmax=bmax, grid_points=steps)

        # uf = float(ms) * np.array([tanh(3*(i-bc)/b_sat) for i in fields]) + hf_sus * fields
        # df = float(ms) * np.array([tanh(3*(i+bc)/b_sat) for i in fields]) + hf_sus * fields
        # irrev_mag = float(ms) * mrs_ms * np.array([cosh(i * (5.5 / b_sat)) ** -1 for i in fields])

        rev_mag = float(ms) * np.array([tanh(2 * i / b_sat) for i in fields]) + hf_sus * fields
        irrev_mag = float(ms) * mrs_ms * np.array([cosh(3.5 * i / b_sat) ** -1 for i in fields])

        if noise:
            noise = max(max(rev_mag), max(irrev_mag)) * noise /100
            rev_mag += np.random.normal(0, noise, len(rev_mag))
            irrev_mag += np.random.normal(0, noise, len(irrev_mag))

        data['down_field'] = RockPyData(column_names=['field', 'mag'], data=np.c_[fields, rev_mag + irrev_mag])
        data['up_field'] = RockPyData(column_names=['field', 'mag'], data=np.c_[fields, rev_mag - irrev_mag])

        return cls(sobj, fpath=None, mdata=data, ftype='simulation',
                   color=color, marker=marker, linestyle=linestyle,
                   idx=idx)

    @classmethod
    def get_grid(cls, bmax=1, grid_points=30, tuning=10):
        grid = []
        # calculating the grid
        for i in range(-grid_points, grid_points + 1):
            if i != 0:
                boi = (abs(i) / i) * (bmax / tuning) * ((tuning + 1) ** (abs(i) / float(grid_points)) - 1.)
            else:  # catch exception for i = 0
                boi = 0
            grid.append(boi)
        return np.array(grid)

    @staticmethod
    def approach2sat_func(h, ms, chi, alpha):
        """
        General approach to saturation function

        Parameters
        ----------
           x: ndarray
              field
           ms: float
              saturation magnetization
           chi: float
              susceptibility
           alpha: float
           beta: float
              not fitted assumed -2

        Returns
        -------
           ndarray:
              :math:`M_s \chi * B + \\alpha * B^{\\beta = -2}`
        """
        return ms + chi * h - alpha * h ** -2  # beta

    @staticmethod
    def fit_tanh(params, x, data=0):
        """
        Function for fitting up to four tanh functions to reversible branch of hysteresis

        Parameters
        ----------
           params: Parameterclass lmfit
              Bti: saturation magnetization of ith component
              Gti: curvature of ith component related to coercivity
           x: array-like
              x-values of data for fit
           data: array-like
              y-values of data for fit

        Returns
        -------
           residual: array-like
              Residual of fitted data and measured data
        """
        Bt1 = params['Bt1'].value
        Bt2 = params['Bt2'].value
        Bt3 = params['Bt3'].value
        Bt4 = params['Bt4'].value
        Gt1 = params['Gt1'].value
        Gt2 = params['Gt2'].value
        Gt3 = params['Gt3'].value
        Gt4 = params['Gt4'].value
        Et = params['Et'].value

        model = Bt1 * np.tanh(Gt1 * x)
        model += Bt2 * np.tanh(Gt2 * x)
        model += Bt3 * np.tanh(Gt3 * x)
        model += Bt4 * np.tanh(Gt4 * x)
        model += Et * x
        return np.array(model - data)

    @staticmethod
    def fit_sech(params, x, data=0):
        """
        Function for fitting up to four sech functions to reversible branch of hysteresis

        Parameters
        ----------
           params: Parameterclass lmfit
              Bsi: saturation magnetization of ith component
              Gsi: curvature of ith component related to coercivity
           x: array-like
              x-values of data for fit
           data: array-like
              y-values of data for fit

        Returns
        -------
           residual: array-like
              Residual of fitted data and measured data
        """
        Bs1 = params['Bs1'].value
        Bs2 = params['Bs2'].value
        Bs3 = params['Bs3'].value
        Bs4 = params['Bs4'].value
        Gs1 = params['Gs1'].value
        Gs2 = params['Gs2'].value
        Gs3 = params['Gs3'].value
        Gs4 = params['Gs4'].value

        model = Bs1 * np.cosh(Gs1 * x) ** -1
        model += Bs2 * np.cosh(Gs2 * x) ** -1
        model += Bs3 * np.cosh(Gs3 * x) ** -1
        model += Bs4 * np.cosh(Gs4 * x) ** -1
        return np.array(model - data)

    @staticmethod
    def unvary_params(params):
        for p in params:
            p.set(vary=False)
            p.set(value=0)
        return params

    @property
    def data(self):
        if not self._data:
            self._data = deepcopy(self._raw_data)
        return self._data

    ####################################################################################################################
    """ formatting functions """
    # have to return mdata matching the measurement type
    @staticmethod
    def format_vsm(ftype_data, sobj_name=None):
        header = ftype_data.header
        segments = ftype_data.get_segments_from_data()
        data = ftype_data.get_data()

        mdata = {}

        if 'adjusted field' in header:
            header[header.index('adjusted field')] = 'field'
            header[header.index('field')] = 'uncorrected field'

        if 'adjusted moment' in header:
            header[header.index('moment')] = 'uncorrected moment'
            header[header.index('adjusted moment')] = 'moment'

        if len(segments['segment number']) == 3:
            mdata.setdefault('virgin', RockPyData(column_names=header, data=data[0],
                                                  units=ftype_data.units).sort('field'))
            mdata.setdefault('down_field', RockPyData(column_names=header, data=data[1],
                                                      units=ftype_data.units).sort('field'))
            mdata.setdefault('up_field', RockPyData(column_names=header, data=data[2],
                                                    units=ftype_data.units).sort('field'))

        if len(segments['segment number']) == 2:
            mdata.setdefault('virgin', None)
            mdata.setdefault('down_field', RockPyData(column_names=header, data=data[0],
                                                      units=ftype_data.units).sort('field'))
            mdata.setdefault('up_field', RockPyData(column_names=header, data=data[1],
                                                    units=ftype_data.units).sort('field'))

        if len(segments['segment number']) == 1:
            mdata.setdefault('virgin', RockPyData(column_names=header, data=data[0],
                                                  units=ftype_data.units
                                                  ).sort('field'))
            mdata.setdefault('down_field', None)
            mdata.setdefault('up_field', None)

        with RockPy3.ignored(AttributeError):
            mdata['virgin'].rename_column('moment', 'mag')

        with RockPy3.ignored(AttributeError):
            mdata['up_field'].rename_column('moment', 'mag')

        with RockPy3.ignored(AttributeError):
            mdata['down_field'].rename_column('moment', 'mag')

        return mdata

    @staticmethod
    def format_mpms(ftype_data, sobj_name=None):

        #get the index of the field column
        field_idx = ftype_data.header.index('Field')
        # get the differences in fieldbetween steps
        diff_data = np.diff(ftype_data.data[:,1])
        # assuming the measurement starts with the down field branch -> the difference between field steps is negative
        if diff_data[0]<0:
            start_idx = min(i+1 for i,v in enumerate(np.diff(ftype_data.data[:,1])) if v > 0)
            df_data = ftype_data.data[:start_idx,:]
            uf_data = ftype_data.data[start_idx-1:,:]
        else:
            raise NotImplementedError('implement hysteresis starting from 0 or up field')

        # set data structure
        mdata = {}
        mdata.setdefault('virgin', None)
        mdata.setdefault('down_field', RockPyData(column_names=[i.lower() for i in ftype_data.header],
                                                  data=df_data, units=ftype_data.units).sort('field'))
        mdata.setdefault('up_field', RockPyData(column_names=[i.lower() for i in ftype_data.header],
                                                  data=uf_data, units=ftype_data.units).sort('field'))

        for dtype in mdata:
            if mdata[dtype]:
                mdata[dtype].rename_column('long moment', 'mag')
                mdata[dtype].rename_column('temperature', 'temp')
        # print(mdata['down_field'])
        return mdata
    @property
    def max_field(self):
        fields = set(self.data['down_field']['field'].v) | set(self.data['up_field']['field'].v)
        return max(fields)

    # ## calculations

    """ RESULT / CALCULATE METHODS """
    ####################################################################################################################
    """ BC """

    @calculate
    def calculate_bc_LINEAR(self, no_points=4, check=False, **non_method_parameters):
        """
        Calculates the coercivity using a linear interpolation between the points crossing the x axis for upfield and down field slope.

        Parameters
        ----------
            field_limit: float
                default: 0, 0mT
                the maximum/ minimum fields used for the linear regression

        Note
        ----
            Uses scipy.linregress for calculation
        """
        # initialize result
        result = []
        # get magneization limits for a calculation using the 2 points closest to 0 fro each direction
        df_moment = sorted(abs(self.data['down_field']['mag'].v))[no_points - 1]
        uf_moment = sorted(abs(self.data['up_field']['mag'].v))[no_points - 1]

        # filter data for fields higher than field_limit
        down_f = self.data['down_field'].filter(abs(self.data['down_field']['mag'].v) <= df_moment)
        up_f = self.data['up_field'].filter(abs(self.data['up_field']['mag'].v) <= uf_moment)

        # calculate bc for both measurement directions
        for i, dir in enumerate([down_f, up_f]):
            slope, intercept, r_value, p_value, std_err = stats.linregress(dir['field'].v, dir['mag'].v)
            result.append(abs(intercept / slope))
            # check plot
            if check:
                x = dir['field'].v
                y_new = slope * x + intercept
                plt.plot(dir['field'].v, dir['mag'].v, '.', color=RockPy3.colorscheme[i])
                plt.plot(x, y_new, color=RockPy3.colorscheme[i])

        # check plot
        if check:
            plt.plot([-np.nanmean(result), np.nanmean(result)], [0, 0], 'xk')
            plt.grid()
            plt.show()

        self.results['bc'] = [[(np.nanmean(result), np.nanstd(result))]]

    @calculate
    def calculate_bc_NONLINEAR(self, no_points=4, check=False, **non_method_parameters):
        """
        Calculates the coercivity using a spline interpolation between the points crossing the x axis for upfield and down field slope.

        Parameters
        ----------
            field_limit: float
                default: 0, 0mT
                the maximum/ minimum fields used for the linear regression

        Note
        ----
            Uses scipy.linregress for calculation
        """
        # initialize result
        result = []
        # get limits for a calculation using the no_points points closest to 0 fro each direction
        df_limit = sorted(abs(self.data['down_field']['field'].v))[no_points - 1]
        uf_limit = sorted(abs(self.data['up_field']['field'].v))[no_points - 1]

        # the field_limit has to be set higher than the lowest field
        # if not the field_limit will be chosen to be 2 points for uf and df separately
        if no_points < 2:
            self.log.warning('NO_POINTS INCOMPATIBLE minimum 2 required' % (no_points))
            self.log.warning('\t\t setting NO_POINTS - << 2 >> ')
            self.calculation_parameter['bc']['no_points'] = 2

        # filter data for fields higher than field_limit
        down_f = self.data['down_field'].filter(abs(self.data['down_field']['field'].v) <= df_limit)
        up_f = self.data['up_field'].filter(abs(self.data['up_field']['field'].v) <= uf_limit)

        x = np.linspace(-min([df_limit, uf_limit]), min([df_limit, uf_limit]), 1000)
        for i, dir in enumerate([down_f, up_f]):
            spl = UnivariateSpline(dir['field'].v, dir['mag'].v)
            y_new = spl(x)
            idx = np.argmin(abs(y_new))
            result.append(abs(x[idx]))

            if check:
                plt.plot(dir['field'].v, dir['mag'].v, '.', color=RockPy3.colorscheme[i])
                plt.plot(x, y_new, color=RockPy3.colorscheme[i])

        if check:
            plt.plot([-np.nanmean(result), np.nanmean(result)], [0, 0], 'xk')
            plt.grid()
            plt.show()

        # set result so it can be accessed
        self.results['bc'] = [[(np.nanmean(result), np.nanstd(result))]]

    @result
    def result_bc(self, recipe='LINEAR', recalc=False, **non_method_parameters):
        """
        Calculates :math:`B_c` using a linear interpolation between the points closest to zero.

           recalc: bool
                     Force recalculation if already calculated
        """
        pass

    ####################################################################################################################
    """ MRS """

    @calculate
    def calculate_mrs(self, no_points=4, check=False, **non_method_parameters):
        # initialize result
        result = []

        # get field limits for a calculation using the 2 points closest to 0 fro each direction
        df_field = sorted(abs(self.data['down_field']['field'].v))[no_points - 1]
        uf_field = sorted(abs(self.data['up_field']['field'].v))[no_points - 1]

        # filter data for fields higher than field_limit
        down_f = self.data['down_field'].filter(abs(self.data['down_field']['field'].v) <= df_field)
        up_f = self.data['up_field'].filter(abs(self.data['up_field']['field'].v) <= uf_field)

        for i, dir in enumerate([down_f, up_f]):
            slope, intercept, r_value, p_value, std_err = stats.linregress(dir['field'].v, dir['mag'].v)
            result.append(abs(intercept))
            # check plot
            if check:
                x = dir['field'].v
                y_new = slope * x + intercept
                plt.plot(dir['field'].v, dir['mag'].v, '.', color=Hysteresis.colors[i])
                plt.plot(x, y_new, color=Hysteresis.colors[i])

        # check plot
        if check:
            plt.plot([0, 0], [-np.nanmean(result), np.nanmean(result)], 'xk')
            plt.grid()
            plt.show()

        self.results['mrs'] = [[(np.nanmean(result), np.nanstd(result))]]

    @result
    def result_mrs(self, recalc=False, **non_method_parameters):
        """
        The default definition of magnetic remanence is the magnetization remaining in zero field after a large magnetic field is applied (enough to achieve saturation).[Banerjee, S. K.; Mellema, J. P. (1974)] The effect of a magnetic hysteresis loop is measured using instruments such as a vibrating sample magnetometer; and the zero-field intercept is a measure of the remanence. In physics this measure is converted to an average magnetization (the total magnetic moment divided by the volume of the sample) and denoted in equations as Mr. If it must be distinguished from other kinds of remanence it is called the saturation remanence or saturation isothermal remanence (SIRM) and denoted by :math:`M_{rs}`.
        """
        pass

    ####################################################################################################################
    """ MS """

    def get_df_uf_plus_minus(self, saturation_percent, ommit_last_n):
        """
        Filters the data :code:`down_field`, :code:`up_field` to be larger than the saturation_field, filters the last :code:`ommit_last_n` and splits into pos and negative components
        """
        # transform from percent value
        saturation_percent /= 100

        # filter ommitted points
        df = self.data['down_field'].filter_idx(
            [i for i in range(len(self.data['down_field']['field'].v))
             if ommit_last_n - 1 < i < len(self.data['down_field']['field'].v) - ommit_last_n])
        uf = self.data['up_field'].filter_idx(
            [i for i in range(len(self.data['up_field']['field'].v))
             if ommit_last_n - 1 < i < len(self.data['up_field']['field'].v) - ommit_last_n])

        # filter for field limits
        df_plus = df.filter(df['field'].v >= saturation_percent * self.max_field)
        df_minus = df.filter(df['field'].v >= saturation_percent * self.max_field)

        uf_plus = uf.filter(uf['field'].v >= saturation_percent * self.max_field)
        uf_minus = uf.filter(uf['field'].v >= saturation_percent * self.max_field)

        return df_plus, df_minus, uf_plus, uf_minus

    @calculate
    def calculate_ms_APP2SAT(self, saturation_percent=75., ommit_last_n=0, check=False, **non_method_parameters):
        """
        Calculates the high field susceptibility using approach to saturation
        :return:
        """

        ms, slope, alpha = self.calc_approach2sat(saturation_percent=saturation_percent,
                                                  ommit_last_n=ommit_last_n)

        self.results['ms'] = [[[np.mean(ms), np.std(ms)]]]
        self.results['hf_sus'] = [[[np.mean(slope), np.std(slope)]]]
        self.results = self.results.append_columns(column_names=['alpha'],
                                                   data=[[[np.mean(slope), np.std(slope)]]])

    @calculate
    def calculate_ms_SIMPLE(self, saturation_percent=75., ommit_last_n=0, check=False, **non_method_parameters):
        """
        Calculates High-Field susceptibility using a simple linear regression on all branches

        Parameters
        ----------
            saturation_percent: float
                default: 75.0
                Defines the field limit in percent of max(field) at which saturation is assumed.
                 e.g. max field : 1T -> saturation asumed at 750mT
            ommit_last_n: int, pos
                last n points of each branch are not used for the calculation

        Calculation
        -----------
            calculates the slope using SciPy.linregress for each branch at positive and negative fields. Giving four values for the slope. The result is the mean for all four values, and the error is the standard deviation

        """

        # initialize result
        hf_sus_result = []
        ms_result = []

        if saturation_percent >= 100:
            self.log.warning('SATURATION > 100%! setting to default value (75%)')
            saturation_percent = 75.0

        # transform from percent value
        saturation_percent /= 100

        # filter ommitted points
        df = self.data['down_field'].filter_idx(
            [i for i in range(len(self.data['down_field']['field'].v))
             if ommit_last_n - 1 < i < len(self.data['down_field']['field'].v) - ommit_last_n])
        uf = self.data['up_field'].filter_idx(
            [i for i in range(len(self.data['up_field']['field'].v))
             if ommit_last_n - 1 < i < len(self.data['up_field']['field'].v) - ommit_last_n])

        # filter for field limits
        df_plus = df.filter(df['field'].v >= saturation_percent * self.max_field)
        df_minus = df.filter(df['field'].v >= saturation_percent * self.max_field)

        uf_plus = uf.filter(uf['field'].v >= saturation_percent * self.max_field)
        uf_minus = uf.filter(uf['field'].v >= saturation_percent * self.max_field)

        for i, dir in enumerate([df_plus, df_minus, uf_plus, uf_minus]):
            slope, intercept, r_value, p_value, std_err = stats.linregress(dir['field'].v, dir['mag'].v)
            hf_sus_result.append(abs(slope))
            ms_result.append(abs(intercept))
            # check plot
            if check:
                x = np.linspace(0, self.max_field)
                y_new = slope * x + intercept
                plt.plot(abs(df['field'].v), abs(df['mag'].v), '-', color=Hysteresis.colors[i])
                plt.plot(x, y_new, '--', color=Hysteresis.colors[i])

        # check plot
        if check:
            plt.plot([0, 0], [-np.nanmean(hf_sus_result), np.nanmean(hf_sus_result)], 'xk')
            plt.grid()
            plt.show()

        self.results['hf_sus'] = [[(np.nanmean(hf_sus_result), np.nanstd(hf_sus_result))]]
        self.results['ms'] = [[(np.nanmean(ms_result), np.nanstd(ms_result))]]

    def calculate_hf_sus_APP2SAT(self, **parameter):
        """
        Calculates the high field susceptibility using approach to saturation
        :return:
        """
        saturation_percent = parameter.get('saturation_percent', 75.)
        remove_last = parameter.get('remove_last', 0)

        res_uf1, res_uf2 = self.calc_approach2sat(saturation_percent=saturation_percent, branch='up_field',
                                                  remove_last=remove_last)
        res_df1, res_df2 = self.calc_approach2sat(saturation_percent=saturation_percent, branch='down_field',
                                                  remove_last=remove_last)
        res = np.c_[res_uf1, res_uf2, res_df1, res_df2]

        self.results['ms'] = [[[np.mean(res, axis=1)[0], np.std(res, axis=1)[0]]]]
        self.results['hf_sus'] = [[[np.mean(res, axis=1)[1], np.std(res, axis=1)[1]]]]
        self.results = self.results.append_columns(column_names=['alpha'],
                                                   data=[[[np.mean(res, axis=1)[2], np.std(res, axis=1)[2]]]])

        parameter.update(dict(method='app2sat'))
        self.calculation_parameter['hf_sus'].update(parameter)
        return self.results['hf_sus'].v[0]

    @result
    def result_ms(self, recipe='SIMPLE', saturation_percent=75., ommit_last_n=0, recalc=False, **non_method_parameters):
        """
        calculates the Ms value with a linear fit

        :Parameters:

           recalc: str standard(False)
              if True result will be forced to be recalculated


           parameter:
              - from_field : field value in % of max. field above which slope seems linear

        :Return:

            RockPyData

        :Methods:

           auto:
             uses simple method

           simple:
              Calculates a simple linear regression of the high field magnetization. The y-intercept is Ms.

           approach_to_sat:
              Calculates a simple approach to saturation :cite:`Dobeneck1996a`

        """
        pass

    @result
    def result_hf_sus(self, recipe='SIMPLE', calculation_method='ms', recalc=False, **non_method_parameters):
        """
        Calculates the result for high field susceptibility using the specified method

        Parameters
        ----------
           saturation_percent: float
              field at which saturation is assumed. Only data at higher (or lower negative) is used for analysis
           method: str
              method of analysis
              :simple: uses simple linear regression above saturation_percent
           recalc: bool
              forced recalculation of result
           options: dict
              additional arguments

        Returns
        -------
           RockPyData
        """
        pass

    # ####################################################################################################################
    # ''' Brh'''
    #
    # @calculate
    # def calculate_brh(self, **parameter):
    #     pass  # todo implement
    #
    # @result
    # def result_brh(self, recalc=False, **options):
    #     """
    #     By definition, Brh is the median destructive field of the vertical hysteresis difference:
    #
    #     .. math::
    #
    #        M_{rh}(B) = \\frac{M^+(B)-M^-(B)}{2}
    #
    #     """
    #     self.calc_result(dict(), recalc)
    #     return self.results['brh']
    #
    # ####################################################################################################################
    # ''' E_delta_t'''
    #
    # @calculate
    # def calculate_e_delta_t(self, **parameter):
    #     """
    #     Method calculates the :math:`E^{\Delta}_t` value for the hysteresis.
    #     It uses scipy.integrate.simps for calculation of the area under the down_field branch for positive fields and
    #     later subtracts the area under the Msi curve.
    #
    #     The energy is:
    #
    #     .. math::
    #
    #        E^{\delta}_t = 2 \int_0^{B_{max}} (M^+(B) - M_{si}(B)) dB
    #
    #     """
    #     # if not self.msi_exists:
    #     #     self.log.error(
    #     #         '%s\tMsi branch does not exist or not properly saturated. Please check datafile' % self.sobj.name)
    #     #     self.results['e_delta_t'] = np.nan
    #     #     return np.nan
    #     #
    #     # # getting data for positive down field branch
    #     # df_pos_fields = [v for v in self.data['down_field']['field'].v if v >= 0] + [0.0]  # need to add 0 to fields
    #     # df_pos = self.data['down_field'].interpolate(df_pos_fields)  # interpolate value for 0
    #     # df_pos_area = abs(sp.integrate.simps(y=df_pos['mag'].v, x=df_pos['field'].v))  # calculate area under downfield
    #     #
    #     # msi_area = abs(sp.integrate.simps(y=self.data['virgin']['mag'].v,
    #     #                                   x=self.data['virgin']['field'].v))  # calulate area under virgin
    #     #
    #     # self.results['e_delta_t'] = abs(2 * (df_pos_area - msi_area))
    #     # self.calculation_parameter['e_delta_t'].update(parameter)
    #     # return self.results['e_delta_t'].v[0]
    #     pass
    #
    # @result
    # def result_e_delta_t(self, recalc=False, **options):
    #     self.calc_result(dict(), recalc)
    #     return self.results['e_delta_t']
    #
    # ####################################################################################################################
    # ''' E_hys'''
    #
    # @calculate
    # def calculate_e_hys(self, **parameter):
    #     '''
    #     Method calculates the :math:`E^{Hys}` value for the hysteresis.
    #     It uses scipy.integrate.simps for calculation of the area under the down_field branch and
    #     later subtracts the area under the up-field branch.
    #
    #     The energy is:
    #
    #     .. math::
    #
    #        E^{Hys} = \int_{-B_{max}}^{B_{max}} (M^+(B) - M^-(B)) dB
    #
    #     '''
    #
    #     df_area = sp.integrate.simps(y=self.data['down_field']['mag'].v,
    #                                  x=self.data['down_field']['field'].v)  # calulate area under down_field
    #     uf_area = sp.integrate.simps(y=self.data['up_field']['mag'].v,
    #                                  x=self.data['up_field']['field'].v)  # calulate area under up_field
    #
    #     self.results['e_hys'] = abs(df_area - uf_area)
    #
    # @result
    # def result_e_hys(self, recalc=False, **options):
    #     self.calc_result(dict(), recalc)
    #     return self.results['e_hys']

    ####################################################################################################################
    ''' Mrs/Ms'''

    @calculate
    def calculate_mrs_ms(self, **non_method_parameters):
        '''
        Method calculates the :math:`E^{Hys}` value for the hysteresis.
        It uses scipy.integrate.simps for calculation of the area under the down_field branch and
        later subtracts the area under the up-field branch.

        The energy is:

        .. math::

           E^{Hys} = \int_{-B_{max}}^{B_{max}} (M^+(B) - M^-(B)) dB

        '''
        mrs = self.result_mrs(**non_method_parameters)
        ms = self.result_ms(**non_method_parameters)
        self.results['mrs_ms'] = [[[mrs[0] / ms[0], mrs[1] + ms[1]]]]

    @result
    def result_mrs_ms(self, recalc=False, **options):
        pass

    # ####################################################################################################################
    # ''' Moment at Field'''
    #
    # @calculate
    # def calculate_m_b(self, b=300., branches='all', **non_method_parameters):
    #     '''
    #
    #     Parameters
    #     ----------
    #         b: field in mT where the moment is returned
    #     '''
    #     aux = []
    #     dtypes = []
    #
    #     possible = {'down_field': 1, 'up_field': -1}
    #
    #     if branches == 'all':
    #         branches = possible
    #     else:
    #         branches = RockPy3.utils.general.to_list(branches)
    #
    #     if any(branch not in possible for branch in branches):
    #         self.log.error('ONE or MORE branches not possible << %s >>' % branches)
    #
    #     for branch in branches:
    #         if self.data[branch]:
    #             field = float(b) * possible[branch]
    #             m = self.data[branch].interpolate(new_variables=field / 1000.)  # correct mT -> T
    #             aux.append(m['mag'].v[0])
    #             dtypes.append(branch)
    #     self.log.info('M(%.1f mT) calculated as mean of %s branch(es)' % (b, dtypes))
    #     self.results['m_b'] = [[[np.nanmean(np.fabs(aux)), np.nanstd(np.fabs(aux))]]]
    #
    # @result
    # def result_m_b(self, recalc=False, **options):
    #     pass

    ####################################################################################################################
    ''' Bcr/ Bc '''

    @calculate
    def calculate_bcr_bc(self,
                         coe_obj=None, bcr_recipe='LINEAR', bcr_no_points=4,
                         bc_no_points=4, bc_recipe='LINEAR',
                         **non_method_parameters):
        '''

        Parameters
        ----------
            b: field in mT where the moment is returned
        '''

        if not coe_obj:
            self.log.info('NO backfield/coe measurement specified: searching through sample')
            coe_objs = [m for m in self.sobj.get_measurement(mtype='backfield') if m.series == self.series]
            if len(coe_objs) == 0:
                self.log.error('CANT find measurement with << backfield, %s >>' % self.stype_sval_tuples)
                return
            elif len(coe_objs) == 1:
                self.log.info('FOUND exactly one measurement with << backfield, %s >>' % self.stype_sval_tuples)
                coe_obj = coe_objs[0]
            else:
                self.log.info('MULTIPLE find backfield/coe measurement found with same stypes/svals using first')
                coe_obj = coe_objs[0]

        bcr = coe_obj.result_bcr(recipe=bcr_recipe, no_points=bcr_no_points, **non_method_parameters)
        bc = self.result_bc(recipe=bc_recipe, no_points=bc_no_points, **non_method_parameters)
        self.results['bcr_bc'] = [[[bcr[0] / bc[0], bcr[1] + bc[1]]]]

    @result
    def result_bcr_bc(self, secondary='backfield', recalc=False, **options):
        pass

    """ CALCULATIONS """

    def get_irreversible(self, correct_symmetry=True):
        """
        Calculates the irreversible hysteretic components :math:`M_{ih}` from the data.

        .. math::

           M_{ih} = (M^+(H) + M^-(H)) / 2

        where :math:`M^+(H)` and :math:`M^-(H)` are the upper and lower branches of the hysteresis loop

        Returns
        -------
           Mih: RockPyData

        """
        field_data = sorted(list(set(self.data['down_field']['field'].v) | set(self.data['up_field']['field'].v)))
        uf = self.data['up_field'].interpolate(field_data)
        df = self.data['down_field'].interpolate(field_data)
        M_ih = deepcopy(uf)
        M_ih['mag'] = (df['mag'].v + uf['mag'].v) / 2

        if correct_symmetry:
            M_ih_pos = M_ih.filter(M_ih['field'].v >= 0).interpolate(np.fabs(field_data))
            M_ih_neg = M_ih.filter(M_ih['field'].v <= 0).interpolate(np.fabs(field_data))

            mean_data = np.mean(np.c_[M_ih_pos['mag'].v, -M_ih_neg['mag'].v], axis=1)
            M_ih['mag'] = list(-mean_data).extend(list(mean_data))

        return M_ih.filter(~np.isnan(M_ih['mag'].v))

    def get_reversible(self):
        """
        Calculates the reversible hysteretic components :math:`M_{rh}` from the data.

        .. math::

           M_{ih} = (M^+(H) - M^-(H)) / 2

        where :math:`M^+(H)` and :math:`M^-(H)` are the upper and lower branches of the hysteresis loop

        Returns
        -------
           Mrh: RockPyData

        """
        field_data = sorted(list(set(self.data['down_field']['field'].v) | set(self.data['up_field']['field'].v)))
        uf = self.data['up_field'].interpolate(field_data)
        df = self.data['down_field'].interpolate(field_data)
        M_rh = deepcopy(uf)
        M_rh['mag'] = (df['mag'].v - uf['mag'].v) / 2
        return M_rh.filter(~np.isnan(M_rh['mag'].v))

    def get_interpolated(self, branch):
        """
        calculates interpolated mag values for all field values for up and down field branch
        :param branch:
        :return:
        """
        field_data = sorted(list(set(self.data['down_field']['field'].v) | set(self.data['up_field']['field'].v)))

        if branch == 'down_field':
            return self.data['down_field'].interpolate(field_data)
        if branch == 'up_field':
            return self.data['up_field'].interpolate(field_data)
        if branch == 'all':
            return self.data['down_field'].interpolate(field_data), self.data['up_field'].interpolate(field_data)

    """ CORRECTIONS """

    def correct_outliers(self, threshold=4, check=False):
        """
        Method that corrects outliers

        Parameters
        ----------
           threshold: int
              standard deviation away from fit

        """
        raise NotImplementedError

    def correct_vsym(self, method='auto', check=False):
        """
        Correction of horizontal symmetry of hysteresis loop. Horizontal displacement is found by looking for the minimum
         of the absolute magnetization value of the :math:`M_{ih}` curve. The hysteresis is then shifted by the field
         value at this point.

        Parameters
        ----------
           method: str
              for implementation of several methods of calculation
           check: str
              plot to check for consistency
        """

        if check:  # for check plot
            uncorrected_data = deepcopy(self.data)

        pos_max = np.mean([np.max(self.data['up_field']['mag'].v), np.max(self.data['down_field']['mag'].v)])
        neg_min = np.mean([np.min(self.data['up_field']['mag'].v), np.min(self.data['down_field']['mag'].v)])
        correct = (pos_max + neg_min) / 2

        for dtype in self.data:
            self.data[dtype]['mag'] = self.data[dtype]['mag'].v - correct
        self.correction.append('vysm')

        if check:
            self.check_plot(uncorrected_data)

    def correct_hsym(self, method='auto', check=False):
        """
        Correction of horizontal symmetry of hysteresis loop. Horizontal displacement is found by looking for the minimum
         of the absolute magnetization value of the :math:`M_{ih}` curve. The hysteresis is then shifted by the field
         value at this point.

        Parameters
        ----------
           method: str
              for implementation of several methods of calculation
           check: str
              plot to check for consistency
        """

        mir = self.get_irreversible()
        idx = np.nanargmin(abs(mir['mag'].v))
        correct = mir['field'].v[idx] / 2

        if check:  # for check plot
            uncorrected_data = deepcopy(self.data)

        for dtype in self.data:
            self.data[dtype]['field'] = self.data[dtype]['field'].v - correct
        self.correction.append('hysm')
        if check:
            self.check_plot(uncorrected_data)

    @correction
    def correct_paramag(self, saturation_percent=75., method='simple', check=False, **parameter):
        """
        corrects data according to specified method

        Parameters
        ----------
           saturation_percent: float
              default: 75.0
           method: str
              default='simple'
              methods= ...
           check: bool
           parameter: dict

        Returns
        -------

        """

        hf_sus = self.result_hf_sus(method=method)[0]

        if check:
            # make deepcopy for checkplot
            uncorrected_data = deepcopy(self.data)

        for dtype in self.data:
            if not self.data[dtype]:
                continue
            correction = hf_sus * self.data[dtype]['field'].v
            self.data[dtype]['mag'] = self.data[dtype]['mag'].v - correction

        if check:
            ms_all, slope_all = self.fit_hf_slope(saturation_percent=saturation_percent)
            i = 0

            for dtype in ['down_field', 'up_field']:
                b_sat = max(self.data[dtype]['field'].v) * (saturation_percent / 100)
                data_plus = self.data[dtype].filter(self.data[dtype]['field'].v >= b_sat)
                data_minus = self.data[dtype].filter(self.data[dtype]['field'].v <= -b_sat)
                std, = plt.plot(abs(data_plus['field'].v), abs(data_plus['mag'].v))
                plt.plot(abs(data_plus['field'].v), ms_all[i] + abs(data_plus['field'].v) * slope_all[i], '--',
                         color=std.get_color())
                i += 1

                std, = plt.plot(abs(data_minus['field'].v), abs(data_minus['mag'].v))
                plt.plot(abs(data_minus['field'].v), ms_all[i] + abs(data_minus['field'].v) * slope_all[i], '--',
                         color=std.get_color())
                i += 1
            plt.legend(loc='best')
            plt.show()
            self.check_plot(uncorrected_data)

    def correct_hf_sus(self, saturation_percent=75., method='simple', check=False, **non_calculation_parameter):
        """
         Wrapper calls correct_paramag
        """
        self.correct_paramag(saturation_percent=saturation_percent, method=method, check=check)

    @correction
    def correct_center(self, check=False):

        if check:
            uncorrected_data = deepcopy(self.data)

        df = self.data['down_field']
        uf = self.data['up_field']

        uf_rotate = self.rotate_branch(uf)
        df_rotate = self.rotate_branch(df)

        fields = sorted(
            list(set(df['field'].v) | set(uf['field'].v) | set(df_rotate['field'].v) | set(uf_rotate['field'].v)))

        # interpolate all branches and rotations
        df = df.interpolate(fields)
        uf = uf.interpolate(fields)
        df_rotate = df_rotate.interpolate(fields)
        uf_rotate = uf_rotate.interpolate(fields)

        down_field_corrected = deepcopy(df)
        up_field_corrected = deepcopy(uf)
        down_field_corrected['mag'] = (df['mag'].v + uf_rotate['mag'].v) / 2

        up_field_corrected['field'] = - down_field_corrected['field'].v
        up_field_corrected['mag'] = - down_field_corrected['mag'].v

        self.data.update(dict(up_field=up_field_corrected, down_field=down_field_corrected))

        if check:
            self.check_plot(uncorrected_data=uncorrected_data)

    def correct_slope(self):  # todo redundant
        """
        The magnetization curve in this region can be expressed as
        .. math::

           M(B) = Ms + \Chi B + \alpha B^{\beta}

        where :math:`\Chi` is the susceptibility of all dia- and paramagnetic components
        (including the para-effect) and the last term represents an individual approach to saturation law.
        :return:
        """
        # calculate approach to saturation ( assuming beta = -1 ) for upfield/downfield branches with pos / negative field
        # assuming 80 % saturation
        a2s_data = map(self.calc_approach2sat, ['down_field'])  # , 'up_field'])
        a2s_data = np.array(a2s_data)
        a2s_data = a2s_data.reshape(1, 2, 3)[0]

        simple = self.result_paramag_slope()
        # print a2s_data

        popt = np.mean(a2s_data, axis=0)
        ms = np.mean(abs(a2s_data[:, 0]))
        ms_std = np.std(abs(a2s_data[:, 0]))
        chi = np.mean(abs(a2s_data[:, 1])) * np.sign(simple)
        chi_std = np.std(abs(a2s_data[:, 1]))

        alpha = np.mean(abs(a2s_data[:, 2])) * np.sign(a2s_data[0, 2])
        alpha_std = np.std(abs(a2s_data[:, 2]))

        for dtype in self.corrected_data:
            self.corrected_data[dtype]['mag'] = self.corrected_data[dtype]['mag'].v - \
                                                self.corrected_data[dtype]['field'].v * chi
        # print(a2s_data), self.result_paramag_slope()
        return ms, chi, alpha

    def correct_holder(self):
        raise NotImplementedError

    def fit_irrev(self, nfunc, check=False):
        """
        Fitting of the irreversible part of the hysteresis

        :param nfunc:
        :return:
        """
        hf_sus = self.result_hf_sus().v[0]
        ms = self.result_ms().v[0]

        # create a set of Parameters
        params = Parameters()
        params.add('Bt1', value=ms, min=0)
        params.add('Bt2', value=ms, min=0)
        params.add('Bt3', value=ms, min=0)
        params.add('Bt4', value=ms, min=0)

        params.add('Gt1', value=0.1)
        params.add('Gt2', value=0.1)
        params.add('Gt3', value=0.1)
        params.add('Gt4', value=0.1)

        # params.add('Et',   value= 9.81e-5,  vary=False)
        params.add('Et', value=hf_sus, vary=True)

        # set parameters to zero and fix them if less functions are required
        if nfunc < 4:
            self.unvary_params([params['Bt4'], params['Gt4']])
        if nfunc < 3:
            self.unvary_params([params['Bt3'], params['Gt3']])
        if nfunc < 2:
            self.unvary_params([params['Bt2'], params['Gt2']])

        data = self.get_irreversible()
        result = minimize(self.fit_tanh, params, args=(data['field'].v, data['mag'].v))

        if check:
            plt.plot(data['field'].v, data['mag'].v / max(data['mag'].v))
            plt.plot(data['field'].v, (data['mag'].v + result.residual) / max(data['mag'].v))
            plt.plot(data['field'].v, result.residual / max(result.residual))
            plt.show()

            print('FITTING RESULTS FOR IRREVERSIBLE (TANH) %i COMPONENTS' % nfunc)
            report_fit(params)

        return result

    def fit_rev(self, nfunc, check):
        """
        Fitting of the irreversible part of the hysteresis

        :param nfunc:
        :return:
        """
        mrs = self.result_mrs().v[0]
        # create a set of Parameters
        rev_params = Parameters()
        rev_params.add('Bs1', value=mrs, min=0)
        rev_params.add('Bs2', value=mrs, min=0)
        rev_params.add('Bs3', value=mrs, min=0)
        rev_params.add('Bs4', value=mrs, min=0)

        rev_params.add('Gs1', value=1)
        rev_params.add('Gs2', value=1)
        rev_params.add('Gs3', value=1)
        rev_params.add('Gs4', value=1)

        # set parameters to zero and fix them if less functions are required
        if nfunc < 4:
            self.unvary_params([rev_params['Bs4'], rev_params['Gs4']])
        if nfunc < 3:
            self.unvary_params([rev_params['Bs3'], rev_params['Gs3']])
        if nfunc < 2:
            self.unvary_params([rev_params['Bs2'], rev_params['Gs2']])

        data = self.get_reversible()
        result = minimize(self.fit_sech, rev_params, args=(data['field'].v, data['mag'].v))
        if check:
            plt.plot(data['field'].v, data['mag'].v / max(data['mag'].v))
            plt.plot(data['field'].v, (data['mag'].v + result.residual) / max(data['mag'].v))
            plt.plot(data['field'].v, result.residual / max(result.residual))
            plt.show()

            print('FITTING RESULTS FOR REVERSIBLE (SECH) %i COMPONENTS' % nfunc)
            report_fit(rev_params)
        return result

    def fit_hysteresis(self, nfunc=1, correct_symmetry=True, check=False):
        """
        Fitting of hysteresis functions. Removes virgin branch in process.
        :param nfunc:
        :return:
        """
        if check:
            initial_data = deepcopy(self.data)

        # calclate irrev & reversible data
        irrev_data = self.get_irreversible(correct_symmetry=correct_symmetry)
        rev_data = self.get_reversible()

        # calculate fit for each component
        irrev_result = self.fit_irrev(nfunc=nfunc, check=check)
        rev_result = self.fit_rev(nfunc=nfunc, check=check)

        # generate new data
        fields = np.linspace(min(irrev_data['field'].v), max(irrev_data['field'].v), 300)
        irrev_mag = self.fit_tanh(irrev_result.params, fields)
        rev_mag = self.fit_sech(rev_result.params, fields)

        df_data = RockPyData(column_names=['field', 'mag'],
                             data=np.array([[fields[i], irrev_mag[i] + rev_mag[i]] for i in xrange(len(fields))]))
        uf_data = RockPyData(column_names=['field', 'mag'],
                             data=np.array([[fields[i], irrev_mag[i] - rev_mag[i]] for i in xrange(len(fields))]))

        self.data['down_field'] = df_data
        self.data['up_field'] = uf_data
        self.data['virgin'] = None

        if check:
            self.check_plot(uncorrected_data=initial_data)

    ### helper functions
    @property
    def msi_exists(self):
        """
        Checks if Msi branch is present in data by comparing the starting point of the virginbranch with Ms.
        If virgin(0) >= 0.7 * Ms it returns True

        Returns
        -------
           bool
        """
        if self.data['virgin']:
            mrs = self.result_mrs().v
            if abs(self.data['virgin']['mag'].v[0]) >= 0.7 * mrs:
                return True

    def data_gridding(self, method='second', grid_points=20, tuning=1, **parameter):
        """
        Data griding after :cite:`Dobeneck1996a`. Generates an interpolated hysteresis loop with
        :math:`M^{\pm}_{sam}(B^{\pm}_{exp})` at mathematically defined (grid) field values, identical for upper
        and lower branch.

        .. math::

           B_{\text{grid}}(i) = \\frac{|i|}{i} \\frac{B_m}{\lambda} \\left[(\lambda + 1 )^{|i|/n} - 1 \\right]

        Parameters
        ----------

           method: str
              method with which the data is fitted between grid points.

              first:
                  data is fitted using a first order polinomial :math:`M(B) = a_1 + a2*B`
              second:
                  data is fitted using a second order polinomial :math:`M(B) = a_1 + a2*B +a3*B^2`

           parameter: dict
              Keyword arguments passed through

        See Also
        --------
           get_grid
        """

        if len(self.data['down_field']['field'].v) <= 50:
            self.log.warning('Hysteresis branches have less than 50 (%i) points, gridding not possible' % (
                len(self.data['down_field']['field'].v)))
            return

        bmax = max([max(self.data['down_field']['field'].v), max(self.data['up_field']['field'].v)])
        bmin = min([min(self.data['down_field']['field'].v), min(self.data['up_field']['field'].v)])
        bm = max([abs(bmax), abs(bmin)])

        grid = Hysteresis.get_grid(bmax=bm, grid_points=grid_points, tuning=tuning, **parameter)
        # interpolate the magnetization values M_int(Bgrid(i)) for i = -n+1 .. n-1
        # by fitting M_{measured}(B_{experimental}) individually in all intervals [Bgrid(i-1), Bgrid(i+1)]
        # with first or second order polinomials

        def first(x, a, b):
            """
            order one polynomial for fitting
            """
            return a + b * x

        def second(x, a, b, c):
            """
            second order polynomial
            """
            return a + b * x + c * x ** 2

        for dtype in ['down_field', 'up_field', 'virgin']:
            # if dtype == 'virgin':
            #     dtype = [i for i in dtype if i >= 0]

            interp_data = RockPyData(column_names=['field', 'mag'])
            d = self._data[dtype]
            for i in range(1, len(grid) - 1):  # cycle through gridpoints
                idx = [j for j, v in enumerate(d['field'].v) if
                       grid[i - 1] <= v <= grid[i + 1]]  # indices of points within the grid points
                if len(idx) > 0:  # if no points between gridpoints -> no interpolation
                    data = deepcopy(d.filter_idx(idx))
                    try:
                        if method == 'first':
                            popt, pcov = curve_fit(first, data['field'].v, data['mag'].v)
                            mag = first(grid[i], *popt)
                            interp_data = interp_data.append_rows(data=[grid[i], mag])
                        if method == 'second':
                            popt, pcov = curve_fit(second, data['field'].v, data['mag'].v)
                            mag = second(grid[i], *popt)
                            interp_data = interp_data.append_rows(data=[grid[i], mag])
                    except TypeError:
                        self.log.error('Length of data for interpolation < 2')
                        self.log.error(
                            'consider reducing number of points for interpolation or lower tuning parameter')

            if 'temperature' in self.data[dtype].column_names:
                temp = np.mean(self.data[dtype]['temperature'].v)
                std_temp = np.std(self.data[dtype]['temperature'].v)
                temp = np.ones(len(interp_data['mag'].v)) * temp
                interp_data = interp_data.append_columns(column_names='temperature', data=temp)

            self.data.update({dtype: interp_data})

    def rotate_branch(self, branch, data='data'):
        """
        rotates a branch by 180 degrees, by multiplying the field and mag values by -1

        :Parameters:

           data: str
              e.g. data, grid_data, corrected_data

           branch: str or. RockPyData
              up-field or down-field
              RockPyData: will rotate the data
        """
        if isinstance(branch, str):
            data = deepcopy(getattr(self, data)[branch])
        if isinstance(branch, RockPyData):
            data = deepcopy(branch)
        data['field'] = -data['field'].v
        data['mag'] = -data['mag'].v
        return data.sort()

    def set_field_limit(self, field_limit):
        """
        Cuts fields with higer or lower values

        Parameters
        ----------
           field_limit: float
              cut-off field, after which the data is removed from self.data. It is still in self.raw_data
        """
        self.log.debug('FILTERING fields larger than %.2f' % field_limit)
        for dtype in self.data:
            self.data[dtype] = self.data[dtype].filter(abs(self.data[dtype]['field'].v) <= field_limit)

    # ## plotting functions

    def export_vftb(self, folder=None, filename=None):
        from os import path

        abbrev = {'hys': 'hys', 'backfield': 'coe', 'irm_acquisition': 'irm', 'temp_ramp': 'rmp'}

        if not folder:
            folder = RockPy3.join(path.expanduser('~'), 'Desktop')

        if self.get_mtype_prior_to(mtype='mass'):
            mass = self.get_mtype_prior_to(mtype='mass').data['data']['mass'].v[0] * 1e5  # mass converted from kg to mg

        if not filename:
            filename = RockPy3.get_fname_from_info(samplegroup='RockPy', sample_name=self.sobj.name,
                                                   mtype='HYS', ftype=self.ftype, mass=mass, mass_unit='mg')[
                       :-4].replace('.', ',') \
                       + '.' + abbrev[self.mtype]

        line_one = 'name: ' + self.sobj.name + '\t' + 'weight: ' + '%.0f mg' % mass
        line_two = ''
        line_three = 'Set 1:'
        line_four = ' field / Oe	mag / emu / g	temp / centigrade	time / s	std dev / %	suscep / emu / g / Oe'
        field, mag, temp, time, std, sus = [], [], [], [], [], []
        for dtype in ['virgin', 'down_field', 'up_field']:
            if 'field' in self.data[dtype].column_names:
                field.extend(
                    self.data[dtype]['field'].v * 10000)  # converted from tesla to Oe #todo unit check and conversion
            if 'mag' in self.data[dtype].column_names:
                mag.extend(
                    self.data[dtype]['mag'].v / mass)  # converted from tesla to Oe #todo unit check and conversion
            if 'temperature' in self.data[dtype].column_names:
                temp.extend(self.data[dtype]['temperature'].v - 274.15)
            else:
                temp = np.zeros(len(field))
            if 'time' in self.data[dtype].column_names:
                time.extend(self.data[dtype]['time'].v)
            else:
                time = np.zeros(len(field))
            if 'std' in self.data[dtype].column_names:
                std.extend(self.data[dtype]['std'].v)
            else:
                std = np.zeros(len(field))
            if 'sus' in self.data[dtype].column_names:
                sus.extend(self.data[dtype]['sus'].v)
            else:
                sus = np.zeros(len(field))
        data = np.c_[field, mag, temp, time, std, sus]
        data = ['\t'.join(map(str, i)) for i in data]
        data = '\n'.join(data)
        with open(RockPy3.join(folder, filename), 'w+') as f:
            f.writelines(line_one + '\n')
            f.writelines(line_two + '\n')
            f.writelines(line_three + '\n')
            f.writelines(line_four + '\n')
            f.writelines(data)

if __name__ == '__main__':
    Study = RockPy3.RockPyStudy()
    s = Study.add_sample(name='S1')
    hys = s.add_simulation(mtype='hys', ms=1)
    # print(hys.data['down_field'])
    hys.correct_paramag()
    import cProfile
    cProfile.run('hys.calc_all()')
    # for dtype in hys.data:
    #     print(dtype)
    #     print(hys.data[dtype])
    # fig = RockPy3.Figure(fig_input=Study)
    # v = fig.add_visual(visual='hysteresis')
    # v = fig.add_visual(visual='hysteresis', visual_input=hys)
    # fig.show()
#