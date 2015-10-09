__author__ = 'volk'
import logging

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from scipy.interpolate import UnivariateSpline

import RockPy
from RockPy.core.data import RockPyData
from RockPy.core import measurement
from RockPy.core.measurement import calculate, result, correction


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

    def __init__(self, sample_obj,
                 mtype, fpath, ftype,
                 **options):

        super(Backfield, self).__init__(sample_obj,
                                        mtype, fpath, ftype,
                                        **options)

    def format_vftb(self):
        '''
        formats the output from vftb to measurement.data
        :return:
        '''
        data = self.ftype_data.out_backfield()
        header = self.ftype_data.header
        self._raw_data['remanence'] = RockPyData(column_names=header, data=data[0])
        self._raw_data['induced'] = None

    def format_vsm(self):
        """
        formats the vsm output to be compatible with backfield measurements

        A VSM backfield measurement can have several measurements in one file.

            IRM acquisition: (field, mag) if measured, the backfield is the second segment -> index changes
            Backfield: (field, mag) always measured. If only backfield is measured index=0 else index=1
                may include the direct moment (induced) which can be used to calculate ms if it is saturated!
        :return:
        """
        data = self.ftype_data.out_backfield()
        header = self.ftype_data.header

        # check for IRM acquisition -> index is changed
        data_idx = 0
        if self.ftype_data.measurement_header['SCRIPT']['Include IRM?'] == 'Yes':
            data_idx += 1
            self.logger.info('IRM acquisition measured, adding new measurement')
            irm = self.sample_obj.add_measurement(mtype='irm_acquisition', fpath=self.fpath, ftype=self.ftype)
            irm._series = self._series

        # backfield measurement and possible induced moment
        self._raw_data['remanence'] = RockPyData(column_names=['field', 'mag'], data=data[data_idx][:, [0, 1]])
        if self.ftype_data.measurement_header['SCRIPT']['Include direct moment?'] == 'Yes':
            self._raw_data['induced'] = RockPyData(column_names=['field', 'mag'], data=data[data_idx][:, [0, 2]])
        else:
            self._raw_data['induced'] = None

        if self.ftype_data.temperature:
            self.temperature = self.ftype_data.temperature

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
        data = self.data['remanence'].filter(abs(self.data['remanence']['mag'].v) <= limit)#.sort('field')
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

    ####################################################################################################################
    ''' S300 '''

    @calculate
    def calculate_s300_LINEAR(self, no_points=4, check=True, **non_calculation_parameter):
        '''
        S300: :math:`(1 - (M_{300mT} /M_{rs})) / 2`

        :return: result
        '''

        if self.data['remanence']['field'].v.all() < 0.300:
            self.results['s300'] = np.nan
            return

        # get field limits for a calculation using the 2 points closest to 0 fro each direction
        idx = np.argmin(np.fabs(self.data['remanence']['field'].v+0.3))+1

        # filter data for fields higher than field_limit
        data = self.data['remanence'].filter_idx(range(idx-no_points/2, idx+no_points/2))
        slope, intercept, r_value, p_value, std_err = stats.linregress(data['field'].v, data['mag'].v)
        result = abs(slope*-0.3+intercept)

        # check plot
        if check:
            x = data['field'].v
            y_new = slope * x + intercept
            plt.plot(data['field'].v, data['mag'].v, '.', color=RockPy.Measurement.colors[0])
            plt.plot(x, y_new, color=RockPy.Measurement.colors[0])

        # check plot
        if check:
            plt.plot(-0.3, -result, 'xk')
            plt.grid()
            plt.show()

        mrs = self.result_mrs()[0]
        self.results['s300'] = [[(1-(np.nanmean(result)/mrs), np.nan)]]

    @calculate
    def calculate_s300_NONLINEAR(self, no_points=6, check=False, **non_calculation_parameter):
        '''
        S300: :math:`(1 - (M_{300mT} /M_{rs})) / 2`

        :return: result
        '''

        if self.data['remanence']['field'].v.all() < 0.300:
            self.results['s300'] = np.nan
            return

        # get field limits for a calculation using the 2 points closest to 0 fro each direction
        idx = np.argmin(np.fabs(self.data['remanence']['field'].v+0.3))+1

        # filter data for fields higher than field_limit
        data = self.data['remanence'].filter_idx(range(idx-no_points/2, idx+no_points/2))

        x = np.linspace(data['field'].v[0], data['field'].v[-1])
        spl = UnivariateSpline(data['field'].v, data['mag'].v)

        result = abs(spl(-0.3))

        # check plot
        if check:
            x = np.linspace(data['field'].v[0], data['field'].v[-1])
            y_new = spl(x)

            plt.plot(data['field'].v, data['mag'].v, '.', color=RockPy.Measurement.colors[0])
            plt.plot(x, y_new, color=RockPy.Measurement.colors[0])
            plt.plot(-0.3, -result, 'xk')
            plt.grid()
            plt.show()

        mrs = self.result_mrs()[0]
        self.results['s300'] = [[(1-(np.nanmean(result)/mrs), np.nan)]]

    @result
    def result_s300(self, recipe='LINEAR', recalc=False, **non_calculation_parameter):
        pass

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
    def calculate_m_b(self, b=300.,  **non_method_parameters):
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
                m = self.data[dtype].interpolate(new_variables=float(b)/1000.) # correct mT -> T
                aux.append(m['mag'].v[0])
                dtypes.append(dtype)
        self.logger.info('M(%.1f mT) calculated as mean of %s branch(es)' %(b, dtypes))
        self.results['m_b'] = [[[np.nanmean(np.fabs(aux)), np.nanstd(np.fabs(aux))]]]

    @result
    def result_m_b(self, recalc=False, **non_method_parameters):
        pass

    ####################################################################################################################

    def plt_backfield(self):
        plt.plot(self.data['remanence']['field'].v, self.data['remanence']['mag'].v, '.-', zorder=1)
        plt.plot(-self.bcr, 0.0, 'x', color='k')

        if self.data['induced']:
            plt.plot(self._data['induced']['field'].v, self._data['induced']['mag'].v, zorder=1)

        plt.axhline(0, color='#808080')
        plt.axvline(0, color='#808080')
        plt.grid()
        plt.title('Backfield %s' % (self.sample_obj.name))
        plt.xlabel('Field [%s]' % ('T'))  # todo replace with data unit
        plt.ylabel('Moment [%s]' % ('Am2'))  # todo replace with data unit
        plt.show()



def test():
    file = os.path.join(RockPy.test_data_path, 'MUCVSM_test01.coe')
    s = RockPy.Sample(name='test_sample')
    coe = s.add_measurement(fpath=file, ftype='vsm', mtype='backfield')

    # print coe.bcr
    # print coe.result_bcr(recipe='NONLINEAR', no_points=8, check=False)
    # print coe.s300
    # print coe.result_s300(recipe='LINEAR', no_points=4, check=True)
    # print coe.result_s300(recipe='NONLINEAR', no_points=6, check=True)
    print coe.ms
if __name__ == '__main__':
    test()
