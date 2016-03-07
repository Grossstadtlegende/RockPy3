__author__ = 'volk'
import logging
import numpy as np
import os
import scipy as sp
from scipy import stats
from scipy.stats import lognorm
from scipy.interpolate import UnivariateSpline
import RockPy3
from RockPy3.core.data import RockPyData
from RockPy3.core import measurement
from RockPy3.core.measurement import calculate, result, correction
import matplotlib.pyplot as plt


class Backfield(measurement.Measurement):
    logger = logging.getLogger('RockPy.MEASUREMENT.Backfield')
    """
    A Backfield Curve can give information on:
       Bcr: the remanence coercivity

       S300: :math:`(1 - (M_{300mT} /M_{rs})) / 2`

    Bcr is determined by finding the intersection of the linear interpolated measurement data with the axis
    representing zero-magnetization.
    For the calculation of S300, the initial magnetization is used as an approximation of the saturation remanence
    :math:`M_{rs}` and the magnetization at 300mT :math:`M_{300mT}` is determined by linear interpolation of measured
    data.

    Possible data structure::

       self.data: the remanence measurement after the field was applied (normal measurement mode for e.g. VFTB or VSM)


    """
    _standard_parameter = {}

    @staticmethod
    def empty_mdata():
        return dict(data=None)

    @classmethod
    def from_simulation(cls, sobj, idx=None,
                        ms=250., bmax=0.5, E=0.005, G=0.3, steps=20, log_steps=False,
                        noise=None, color=None, marker=None, linestyle=None):
        """
        Simulates a backfield measurement based on a single log-normal gaussian distribution.

        E:  Median destructive field - represents the mean value of the log-Gaussian distribution, and therefore, the
            logarithmic field value of the maximum gradient.

        G:  G describes the standard deviation or half-width of the distribution.

        """
        cls.log.info('CREATING simulation measurement with {}'.format(locals()))

        calculation = np.arange(0, 2, 1e-6)
        fields = np.round(np.linspace(1e-9, bmax, steps), 4)
        dist = lognorm([G], loc=E)
        calc_mag = dist.cdf(calculation)
        indices = [np.argmin(abs(v-calculation)) for v in fields]
        mag = [calc_mag[i] for i in indices]
        # mag /= 1/2 * max(mag)
        # mag -= 1
        # mag *= ms

        # mag = G * (np.sqrt(np.pi) / 2) * (sp.special.erf((log_fields - E) / G) - sp.special.erf((-E) / G)) #leonhardt2004

        # if noise:
        # n = np.random.normal(0, ms * (noise / 100), steps)
        # mag = mag + n

        mdata = cls.empty_mdata()
        mdata['remanence'] = RockPyData(column_names=['field', 'mag'], data=np.c_[fields-bmax, mag])

        return cls(sobj, fpath=None, mdata=mdata, ftype='simulation',
                   color=color, marker=marker, linestyle=linestyle,
                   idx=idx)

    ####################################################################################################################
    """ formatting functions """

    @staticmethod
    def format_vsm(ftype_data, sobj_name=None):
        '''
        formats the output from vftb to measurement.data
        :return:
        '''
        idx = 0
        data = ftype_data.get_data()
        header = ftype_data.header

        # from pprint import pprint
        # pprint(ftype_data.info_header)

        # check if vsm file actually contains a dcd measurement
        if not ftype_data.info_header['include dcd?']:
            return
        else:
            if ftype_data.info_header['include irm?']:
                idx +=1
        mdata = {}
        mdata['data'] = RockPyData(column_names=header, data=data[idx])
        mdata['data'].define_alias('mag', 'remanence')

        return mdata

    @staticmethod
    def format_vftb(ftype_data, sobj_name=None):
        '''
        formats the output from vftb to measurement.data
        :return:
        '''
        data = ftype_data.data
        header = ftype_data.header

        mdata = {}
        mdata['data'] = RockPyData(column_names=header, data=data[0])
        mdata['data'].define_alias('variable', 'field')

        return mdata

    ####################################################################################################################
    ''' Mrs '''

    @calculate
    def calculate_mrs(self, **non_method_parameters):
        """
        Magnetic Moment at last measurement point
        :param parameter:
        :return:
        """
        start = self.data['data']['mag'].v[0]
        end = self.data['data']['mag'].v[-1]
        self.results['mrs'] = [[[np.nanmean([abs(start), abs(end)]), np.nanstd([abs(start), abs(end)])]]]

    @result
    def result_mrs(self, **non_method_parameters):
        pass

    ####################################################################################################################
    ''' Bcr '''

    @calculate
    def calculate_bcr(self, no_points=4, check=False, **non_method_parameters):
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
        moment = sorted(abs(self.data['data']['mag'].v))[no_points - 1]

        # filter data for fields higher than field_limit
        data = self.data['data'].filter(abs(self.data['data']['mag'].v) <= moment)

        # calculate bcr
        slope, intercept, r_value, p_value, std_err = stats.linregress(data['field'].v, data['mag'].v)
        result.append(abs(intercept / slope))
        # check plot
        if check:
            x = data['field'].v
            y_new = slope * x + intercept
            plt.plot(data['field'].v, data['mag'].v, '.', color=Backfield.colors[0])
            plt.plot(x, y_new, color=Backfield.colors[0])

        # check plot
        if check:
            plt.plot([-np.nanmean(result), np.nanmean(result)], [0, 0], 'xk')
            plt.grid()
            plt.show()

        self.results['bcr'] = [[(np.nanmean(result), np.nan)]]

    @calculate
    def calculate_bcr_NONLINEAR(self, no_points=4, check=False, **non_method_parameters):
        """
        Calculates the coercivity of remanence using a spline interpolation between the points crossing
        the x axis for upfield and down field slope.

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
        limit = sorted(abs(self.data['data']['mag'].v))[no_points - 1]
        # the field_limit has to be set higher than the lowest field
        # if not the field_limit will be chosen to be 2 points for uf and df separately
        if no_points < 2:
            self.logger.warning('NO_POINTS INCOMPATIBLE minimum 2 required' % (no_points))
            self.logger.warning('\t\t setting NO_POINTS - << 2 >> ')
            self.calculation_parameter['bcr']['no_points'] = 2

        # filter data for fields higher than field_limit
        data = self.data['data'].filter(abs(self.data['data']['mag'].v) <= limit)  # .sort('field')
        x = np.linspace(data['field'].v[0], data['field'].v[-1])

        spl = UnivariateSpline(data['field'].v, data['mag'].v)
        y_new = spl(x)
        idx = np.argmin(abs(y_new))
        result = abs(x[idx])

        if check:
            plt.plot(data['field'].v, data['mag'].v, '.', color=RockPy.Measurement.colors[0])
            plt.plot(x, y_new, color=RockPy.Measurement.colors[0])
            plt.plot(-result, 0, 'xk')
            plt.grid()
            plt.show()

        # set result so it can be accessed
        self.results['bcr'] = [[(np.nanmean(result), np.nanstd(result))]]

    @result
    def result_bcr(self, recipe='DEFAULT', **non_calculation_parameters):
        """
        calculates :math:`B_{cr}`
        """
        pass

    # ####################################################################################################################
    # ''' S300 '''
    #
    # @calculate
    # def calculate_s300_DEFAULT(self, no_points=4, check=True, **non_calculation_parameter):
    #     '''
    #     S300: :math:`(1 - (M_{300mT} /M_{rs})) / 2`
    #
    #     :return: result
    #     '''
    #
    #     if self.data['data']['field'].v.all() < 0.300:
    #         self.results['s300'] = np.nan
    #         return
    #
    #     # get field limits for a calculation using the 2 points closest to 0 fro each direction
    #     idx = np.argmin(np.fabs(self.data['data']['field'].v+0.3))+1
    #
    #     # filter data for fields higher than field_limit
    #     data = self.data['data'].filter_idx(range(idx-no_points/2, idx+no_points/2))
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(data['field'].v, data['mag'].v)
    #     result = abs(slope*-0.3+intercept)
    #
    #     # check plot
    #     if check:
    #         x = data['field'].v
    #         y_new = slope * x + intercept
    #         plt.plot(data['field'].v, data['mag'].v, '.', color=RockPy.Measurement.colors[0])
    #         plt.plot(x, y_new, color=RockPy.Measurement.colors[0])
    #
    #     # check plot
    #     if check:
    #         plt.plot(-0.3, -result, 'xk')
    #         plt.grid()
    #         plt.show()
    #
    #     mrs = self.result_mrs()[0]
    #     self.results['s300'] = [[(1-(np.nanmean(result)/mrs), np.nan)]]
    #
    # @calculate
    # def calculate_s300_NONLINEAR(self, no_points=6, check=False, **non_calculation_parameter):
    #     '''
    #     S300: :math:`(1 - (M_{300mT} /M_{rs})) / 2`
    #
    #     :return: result
    #     '''
    #
    #     if self.data['data']['field'].v.all() < 0.300:
    #         self.results['s300'] = np.nan
    #         return
    #
    #     # get field limits for a calculation using the 2 points closest to 0 fro each direction
    #     idx = np.argmin(np.fabs(self.data['data']['field'].v+0.3))+1
    #
    #     # filter data for fields higher than field_limit
    #     data = self.data['data'].filter_idx(range(idx-no_points/2, idx+no_points/2))
    #
    #     x = np.linspace(data['field'].v[0], data['field'].v[-1])
    #     spl = UnivariateSpline(data['field'].v, data['mag'].v)
    #
    #     result = abs(spl(-0.3))
    #
    #     # check plot
    #     if check:
    #         x = np.linspace(data['field'].v[0], data['field'].v[-1])
    #         y_new = spl(x)
    #
    #         plt.plot(data['field'].v, data['mag'].v, '.', color=RockPy.Measurement.colors[0])
    #         plt.plot(x, y_new, color=RockPy.Measurement.colors[0])
    #         plt.plot(-0.3, -result, 'xk')
    #         plt.grid()
    #         plt.show()
    #
    #     mrs = self.result_mrs()[0]
    #     self.results['s300'] = [[(1-(np.nanmean(result)/mrs), np.nan)]]
    #
    # @result
    # def result_s300(self, recipe='DEFAULT', recalc=False, **non_calculation_parameter):
    #     pass

    ####################################################################################################################
    #
    # ''' Moment at Field'''
    #
    # @calculate_new
    # def calculate_m_b(self, b=.3, **non_method_parameters):
    #     '''
    #
    #     Parameters
    #     ----------
    #         b: field in mT where the moment is returned
    #     '''
    #     aux = []
    #     dtypes = []
    #     for dtype in ['data']:
    #         if self.data[dtype]:
    #             m = self.data[dtype].interpolate(new_variables=float(b))
    #             aux.append(m['mag'].v[0])
    #             dtypes.append(dtype)
    #     self.logger.info('M(%.1f mT) calculated as mean of %s branch(es)' % (b, dtypes))
    #     self.results['m_b'] = [[[np.nanmean(np.fabs(aux)), np.nanstd(np.fabs(aux))]]]
    #
    # @result_new
    # def result_m_b(self, recalc=False, **non_method_parameters):
    #     pass


def test():
    file = '/Users/mike/Dropbox/experimental_data/COE/FeNiX/FeNiX_FeNi20-G-a-001-M02_COE_VSM#15[mg]_[]_[]#milling time_1_hrs;Ni_20_perc#STD003.001'
    s = RockPy3.Sample(name='test_sample')
    coe = s.add_simulation(mtype='backfield', bmax=1, noise=1)
    # print(coe.data['data'])
    plt.plot(coe.data['data']['field'].v, coe.data['data']['mag'].v)
    plt.show()
    # coe = s.add_measurement(fpath=file, ftype='vsm', mtype='backfield')
    # print(coe.result_s300(recipe='DEFAULT', no_points=4, check=True))
    # print(coe.result_s300(recipe='NONLINEAR', no_points=6, check=True))


if __name__ == '__main__':
    S = RockPy3.Study
    s = S.add_sample(name='test')
    # m = s.add_measurement(fpath='/Users/mike/Dropbox/experimental_data/FeNiX/FeNi20J/FeNi_FeNi20-Jz000-G03_COE_VSM#50,3[mg]_[]_[]#mtime_000_min;GC_03_No;rpm_400_##.001')
    # m = s.add_measurement(fpath='/Users/Mike/Dropbox/experimental_data/001_PintP/LF4C/VFTB/P0-postTT/140310_1a.coe',
    #                         mtype='backfield',
    #                         ftype='vftb')
    m = s.add_measurement(fpath='/Users/Mike/Dropbox/experimental_data/0915-LT_pyrrhtotite/LTPY_P15a_COE_VSM#[]_[]_[]#TEMP_020_K#STD001.001',
                            mtype='backfield',
                            ftype='vsm')
    # fig = RockPy3.Figure(fig_input=S)
    # v = fig.add_visual('resultseries', result='bc', series='mtime', xscale='log')
    # fig.show()
