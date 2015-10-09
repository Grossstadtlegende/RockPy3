__author__ = 'volk'
import matplotlib.pyplot as plt

from RockPy.core import measurement
from RockPy.core.data import RockPyData
import numpy as np
import RockPy

class Acquisition(measurement.Measurement):
    def __init__(self, sample_obj,
                 mtype, fpath, ftype,
                 **options):
        self._data = {'remanence': None,
                      'induced': None}
        super(Acquisition, self).__init__(sample_obj, mtype, fpath, ftype)
        self.variable = 'variable'

    def format_vftb(self):
        data = self.ftype_data.get_data()
        header = self.ftype_data.header
        # self.log.debug('FORMATTING << %s >> raw_data for << VFTB >> data structure' % (self.mtype))
        self._raw_data['remanence'] = RockPyData(column_names=header, data=data[0])

    def format_vsm(self):
        """
        formats the vsm output to be compatible with backfield measurements
        :return:
        """
        data = self.ftype_data.out_backfield()
        header = self.ftype_data.header

        # check for IRM acquisition
        if self.ftype_data.measurement_header['SCRIPT']['Include IRM?'] == 'Yes':
            self._raw_data['remanence'] = RockPyData(column_names=['field', 'mag'], data=data[0][:, [0, 1]])

    def format_cryomag(self):
        # self.log.debug('FORMATTING << %s >> raw_data for << cryomag >> data structure' % (self.mtype))

        data = self.ftype_data.out_trm()
        header = self.ftype_data.float_header
        self._raw_data['remanence'] = RockPyData(column_names=header, data=data)
        self._raw_data['induced'] = None

    def format_jr6(self):
        """
        formats the jr6 data file to be compliant to a thermal acquisition.

            NRM may be measured - NRM of sample
            A90 may be measured - 90mT af demag
        """
        if 'nrm' in self.ftype_data.modes:
            nrm_idx = np.where(self.ftype_data.modes == 'nrm')
        if 'a90' in self.ftype_data.modes:
            a90_idx = np.where(self.ftype_data.modes == 'a90')

        data_idx = np.where(self.ftype_data.modes != 'a90')

        data = RockPyData(column_names=['variable', 'x', 'y', 'z'],
                          data=np.c_[self.ftype_data.modes[data_idx].astype(float),
                                     self.ftype_data.get_data()[data_idx]],
                          units=['', 'A m^2', 'A m^2', 'A m^2'])

        data.define_alias('m', ('x', 'y', 'z'))
        data = data.sort('variable')
        self._raw_data = {'data': data.append_columns('mag', data.magnitude('m'))}

    def plt_data(self):
        std, = plt.plot(self.data['data']['variable'].v, self.data['data']['mag'].v, zorder=1)
        plt.grid()
        plt.axhline(0, color='#808080')
        plt.axvline(0, color='#808080')
        var_idx = self._raw_data['data'].column_names_to_indices('variable')[0]
        plt.xlabel('{} [{:~}]'.format(self.variable, self._raw_data['data'].units[var_idx].units))  # todo data.unit
        data_idx = self._raw_data['data'].column_names_to_indices('mag')[0]
        plt.ylabel('Magnetic Moment {:~}'.format(self._raw_data['data']['mag'].units[0].units))  # todo data.unit
        plt.show()


class Trm_Acquisition(Acquisition):
    def __init__(self, sample_obj,
                 mtype, fpath, ftype,
                 **options):
        self._data = {'data': None,
                      }
        super(Trm_Acquisition, self).__init__(sample_obj, mtype, fpath, ftype)
        self.variable = 'temp'

    def format_jr6(self):
        """
        formats the jr6 data file to be compliant to a thernal acquisition.

            NRM may be measured - NRM of sample
            A90 may be measured - 90mT af demag
        """
        super(Trm_Acquisition, self).format_jr6()
        self._raw_data['data'].rename_column('variable', 'temp')
        self._raw_data['data'].define_alias('variable', 'temp')
        self._raw_data['data'].units[0] = RockPy.core.ureg('degC')
        print self._raw_data['data']['mag'].units

class Irm_Acquisition(Acquisition):

    def plt_irm(self):
        plt.title('IRM acquisition %s' % (self.sample_obj.name))
        std, = plt.plot(self.remanence['field'].v, self.remanence['mag'].v, zorder=1)
        plt.grid()
        plt.axhline(0, color='#808080')
        plt.axvline(0, color='#808080')
        plt.xlabel('Field [%s]' % ('T'))  # todo data.unit
        plt.ylabel('Magnetic Moment [%s]' % ('Am2'))  # todo data.unit
        plt.show()
