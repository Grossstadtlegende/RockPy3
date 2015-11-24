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

       self.remanence: the remanence measurement after the field was applied (normal measurement mode for e.g. VFTB or VSM)
       self.induced: the induced moment measurement while the field is applied (only VSM)

    Notes
    -----
    - VSM- files with irm acquisition will add a new irm_acquisition measurement to the sample

    """
    _standard_parameter = {}

    @staticmethod
    def empty_mdata():
        return dict(remanence=None, induced=None)

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
        data = ftype_data.get_data()
        header = ftype_data.header

        mdata = {'remanence': None, 'induced': None}
        mdata['remanence'] = RockPyData(column_names=header, data=data[1])
        mdata['remanence'].define_alias('mag', 'remanence')

        try:
            mdata['induced'] = RockPyData(column_names=header, data=data[0])
        except:
            RockPy3.logger.error('CANT find induced magnetization column')

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
        start = self.data['remanence']['mag'].v[0]
        end = self.data['remanence']['mag'].v[-1]
        self.results['mrs'] = [[[np.nanmean([abs(start), abs(end)]), np.nanstd([abs(start), abs(end)])]]]

    @result
    def result_mrs(self, recalc=False):
        pass

    ####################################################################################################################
    ''' Bcr '''

    @calculate
    def calculate_bcr_LINEAR(self, no_points=4, check=False, **non_method_parameters):
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
        moment = sorted(abs(self.data['remanence']['mag'].v))[no_points - 1]

        # filter data for fields higher than field_limit
        data = self.data['remanence'].filter(abs(self.data['remanence']['mag'].v) <= moment)

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
        limit = sorted(abs(self.data['remanence']['mag'].v))[no_points - 1]
        # the field_limit has to be set higher than the lowest field
        # if not the field_limit will be chosen to be 2 points for uf and df separately
        if no_points < 2:
            self.logger.warning('NO_POINTS INCOMPATIBLE minimum 2 required' % (no_points))
            self.logger.warning('\t\t setting NO_POINTS - << 2 >> ')
            self.calculation_parameter['bcr']['no_points'] = 2

        # filter data for fields higher than field_limit
        data = self.data['remanence'].filter(abs(self.data['remanence']['mag'].v) <= limit)  # .sort('field')
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
    def result_bcr(self, recipe='LINEAR', recalc=False, **non_calculation_parameters):
        """
        calculates :math:`B_{cr}`
        """
        pass

    # ####################################################################################################################
    # ''' S300 '''
    #
    # @calculate
    # def calculate_s300_LINEAR(self, no_points=4, check=True, **non_calculation_parameter):
    #     '''
    #     S300: :math:`(1 - (M_{300mT} /M_{rs})) / 2`
    #
    #     :return: result
    #     '''
    #
    #     if self.data['remanence']['field'].v.all() < 0.300:
    #         self.results['s300'] = np.nan
    #         return
    #
    #     # get field limits for a calculation using the 2 points closest to 0 fro each direction
    #     idx = np.argmin(np.fabs(self.data['remanence']['field'].v+0.3))+1
    #
    #     # filter data for fields higher than field_limit
    #     data = self.data['remanence'].filter_idx(range(idx-no_points/2, idx+no_points/2))
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
    #     if self.data['remanence']['field'].v.all() < 0.300:
    #         self.results['s300'] = np.nan
    #         return
    #
    #     # get field limits for a calculation using the 2 points closest to 0 fro each direction
    #     idx = np.argmin(np.fabs(self.data['remanence']['field'].v+0.3))+1
    #
    #     # filter data for fields higher than field_limit
    #     data = self.data['remanence'].filter_idx(range(idx-no_points/2, idx+no_points/2))
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
    # def result_s300(self, recipe='LINEAR', recalc=False, **non_calculation_parameter):
    #     pass

    ####################################################################################################################
    ''' Ms '''

    @calculate
    def calculate_ms(self, **non_method_parameters):
        """
                Magnetic Moment at last measurement point
                :param parameter:
                :return:
                """

        if not self.data['induced']:
            return

        ms = self.data['induced']['mag'].v[-1]
        self.results['ms'] = [[[abs(ms), np.nan]]]

    @result
    def result_ms(self, recalc=False):
        pass

    ####################################################################################################################

    ''' Moment at Field'''

    @calculate
    def calculate_m_b(self, b=300., **non_method_parameters):
        '''

        Parameters
        ----------
            b: field in mT where the moment is returned
        '''
        aux = []
        dtypes = []
        for dtype in ['virgin', 'down_field', 'up_field']:
            if not dtype in self.data:
                continue
            if self.data[dtype]:
                m = self.data[dtype].interpolate(new_variables=float(b) / 1000.)  # correct mT -> T
                aux.append(m['mag'].v[0])
                dtypes.append(dtype)
        self.logger.info('M(%.1f mT) calculated as mean of %s branch(es)' % (b, dtypes))
        self.results['m_b'] = [[[np.nanmean(np.fabs(aux)), np.nanstd(np.fabs(aux))]]]

    @result
    def result_m_b(self, recalc=False, **non_method_parameters):
        pass


def test():
    file = '/Users/mike/Dropbox/experimental_data/COE/FeNiX/FeNiX_FeNi20-G-a-001-M02_COE_VSM#15[mg]_[]_[]#milling time_1_hrs;Ni_20_perc#STD003.001'
    s = RockPy3.Sample(name='test_sample')
    coe = s.add_simulation(mtype='backfield', bmax=1, noise=1)
    # print(coe.data['remanence'])
    plt.plot(coe.data['remanence']['field'].v, coe.data['remanence']['mag'].v)
    plt.show()
    # coe = s.add_measurement(fpath=file, ftype='vsm', mtype='backfield')
    # print(coe.result_s300(recipe='LINEAR', no_points=4, check=True))
    # print(coe.result_s300(recipe='NONLINEAR', no_points=6, check=True))


if __name__ == '__main__':
    test()
