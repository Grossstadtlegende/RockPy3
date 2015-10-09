import matplotlib.pyplot as plt
import numpy as np

from RockPy.core.data import RockPyData
from RockPy.core import measurement


class AfDemag(measurement.Measurement):
    '''
    '''

    def __init__(self, sample_obj,
                 mtype, fpath, ftype,
                 mag_method='', demag_type='af3',
                 **options):

        if 'magmethod' in options:
            mag_method = options['magmethod']

        self.demag_type = demag_type
        self.mag_method = mag_method

        super(AfDemag, self).__init__(sample_obj,
                                      mtype, fpath, ftype,
                                      **options)

    def format_jr6(self):
        data = RockPyData(column_names=['field', 'x', 'y', 'z'], data=self.ftype_data.out_afdemag())
        data.define_alias('m', ('x', 'y', 'z'))
        self._raw_data['data'] = data.append_columns('mag', data.magnitude('m'))

    def format_sushibar(self):
        data = RockPyData(column_names=['field', 'x', 'y', 'z'],
                               data=self.ftype_data.out_afdemag())  # , units=['mT', 'Am2', 'Am2', 'Am2'])
        data.define_alias('m', ('x', 'y', 'z'))
        self._raw_data['data'] = data.append_columns('mag', data.magnitude('m'))

    def format_cryomag(self):
        data = RockPyData(column_names=self.ftype_data.float_header,
                               data=self.ftype_data.get_float_data())
        if self.demag_type != 'af3':
            idx = [i for i, v in enumerate(self.ftype_data.steps) if v == self.demag_type]
            data = data.filter_idx(idx)
        data.define_alias('m', ('x', 'y', 'z'))
        self._raw_data['data'] = data.append_columns('mag', data.magnitude('m'))
        self._raw_data['data'].rename_column('step', 'field')

    def format_pmd(self):
        data = RockPyData(column_names=['field', 'x', 'y', 'z'], data=self.ftype_data.out_afdemag())
        data.define_alias('m', ('x', 'y', 'z'))
        self._raw_data['data'] = data.append_columns('mag', data.magnitude('m'))

    def result_mdf(self, component='mag', interpolation='linear', recalc=False):
        """
        Calculates the MDF (median destructive field from data using linear interpolation between closest points

        :param parameter: interpolation:
           interpolation methods:

           'linear' : linear interpolation between points
           'smooth_spline' : calculation using a smoothing spline


        :param recalc:
        :return:
        """
        parameter = {'component': component,
                     'interpolation': interpolation}
        self.calc_result(parameter, recalc)
        return self.results['mdf']

    def calculate_mdf(self, **parameter):

        component = parameter.get('component', 'mag')
        data = self._data['data']  # normalize data #todo replace with normalize function
        data.define_alias('variable', component)
        data_max = max(data[component].v)
        data = data.sort('mag')
        data = data.interpolate([data_max*0.5])
        self.results['mdf'] = data['field'].v

    # ## INTERPOLATION

    def interpolate_smoothing_spline(self, y_component='mag', x_component='field', out_spline=False):
        """
        Interpolates using a smoothing spline between a x and y component
        :param y_component:
        :param x_component:
        :param out_spline:
        :return:
        """
        from scipy.interpolate import UnivariateSpline

        x_old = self._data['data'][x_component].v  #
        y_old = self._data['data'][y_component].v
        smoothing_spline = UnivariateSpline(x_old, y_old, s=1)

        if not out_spline:
            x_new = np.linspace(min(x_old), max(x_old), 100)
            y_new = smoothing_spline(x_new)
            out = RockPyData(column_names=[x_component, y_component], data=np.c_[x_new, y_new])
            return out
        else:
            return smoothing_spline

    def interpolation_spline(self, y_component='mag', x_component='field', out_spline=False):

        from scipy.interpolate import UnivariateSpline

        x_old = self._data['data'][x_component].v  #
        y_old = self._data['data'][y_component].v
        smoothing_spline = UnivariateSpline(x_old, y_old, s=1)

        if not out_spline:
            x_new = np.linspace(min(x_old), max(x_old), 100)
            y_new = smoothing_spline(x_new)
            out = RockPyData(column_names=[x_component, y_component], data=np.c_[x_new, y_new])
            return out
        else:
            return smoothing_spline

    def plt_afdemag(self, norm=False):
        if norm:
            norm_factor = max(self._data['data']['mag'])
        else:
            norm_factor = 1

        plt.title('%s' % self.sample_obj.name)
        plt.plot(self._data['data']['field'].v, self._data['data']['mag'].v / norm_factor, '.-')
        plt.xlabel('field [%s]' % 'mT')
        plt.ylabel('Moment [%s]' % 'Am2')
        plt.grid()
        plt.show()