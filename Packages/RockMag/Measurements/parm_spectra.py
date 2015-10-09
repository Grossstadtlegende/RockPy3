from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from RockPy.core.data import RockPyData
from RockPy.core import measurement


class Parm_Spectra(measurement.Measurement):
    '''
    '''

    def __init__(self, sample_obj,
                 mtype, fpath, ftype,
                 mag_method='',
                 **options):
        super(Parm_Spectra, self).__init__(sample_obj,
                                           mtype, fpath, ftype,
                                           **options)

    def format_sushibar(self):
        data = self.ftype_data.out_parm_spectra()
        self.af3 = RockPyData(column_names=data[1],
                              data=data[0][0],
                              units=data[2])
        self.af3.define_alias('m', ( 'x', 'y', 'z'))
        self.af3 = self.af3.append_columns('mag', self.af3.magnitude('m'))
        self.af3 = self.af3.append_columns(column_names='mean_window',
                                           data=np.array([self.af3['upper_window'].v + self.af3['lower_window'].v])[
                                                    0] / 2)

        self.data = RockPyData(column_names=data[1],
                               data=data[0][1:],
                               units=data[2])
        self.data.define_alias('m', ( 'x', 'y', 'z'))
        self.data = self.data.append_columns('mag', self.data.magnitude('m'))
        self.data = self.data.append_columns(column_names='mean_window',
                                             data=np.array([self.data['upper_window'].v + self.data['lower_window'].v])[
                                                      0] / 2)
        self.data.define_alias('variable', 'mean_window')

    def _get_cumulative_data(self, subtract_af3=True):
        cumulative_data = deepcopy(self.data)

        if subtract_af3:
            cumulative_data['x'] = cumulative_data['x'].v - self.af3['x'].v
            cumulative_data['y'] = cumulative_data['y'].v - self.af3['y'].v
            cumulative_data['z'] = cumulative_data['z'].v - self.af3['z'].v
            cumulative_data['mag'] = cumulative_data.magnitude('m')

        cumulative_data['x'] = [np.sum(cumulative_data['x'].v[:i]) for i,v in enumerate(cumulative_data['x'].v)]
        cumulative_data['y'] = [np.sum(cumulative_data['y'].v[:i]) for i,v in enumerate(cumulative_data['y'].v)]
        cumulative_data['z'] = [np.sum(cumulative_data['z'].v[:i]) for i,v in enumerate(cumulative_data['z'].v)]
        cumulative_data['mag'] = cumulative_data.magnitude('m')

        return cumulative_data

    def plt_parm_spectra(self, subtract_af3=True, rtn=False, norm=False, fill=False):
        plot_data = deepcopy(self.data)

        if subtract_af3:
            plot_data['x'] = plot_data['x'].v - self.af3['x'].v
            plot_data['y'] = plot_data['y'].v - self.af3['y'].v
            plot_data['z'] = plot_data['z'].v - self.af3['z'].v
            plot_data['mag'] = plot_data.magnitude('m')

        if norm:
            norm_factor = max(plot_data['mag'].v)
        else:
            norm_factor = 1

        if fill:
            plt.fill_between(plot_data['mean_window'].v, 0, plot_data['mag'].v / norm_factor,
                             alpha=0.1,
                             label='pARM spetra')
        else:
            plt.plot(plot_data['mean_window'].v, 0, plot_data['mag'].v / norm_factor,
                     label='pARM spetra')
        if not rtn:
            plt.show()


    def plt_parm_acquisition(self, subtract_af3=True, rtn=False, norm=False):
        plot_data = self._get_cumulative_data(subtract_af3=subtract_af3)

        if norm:
            norm_factor = max(plot_data['mag'].v)
        else:
            norm_factor = 1

        plt.plot(plot_data['mean_window'].v, plot_data['mag'].v / norm_factor, label='pARM acquisition')

        if not rtn:
            plt.show()


    def plt_acq_spec(self, subtract_af3=True, norm=True):
        self.plt_parm_spectra(subtract_af3=subtract_af3, rtn=True, norm=norm, fill=True)
        self.plt_parm_acquisition(subtract_af3=subtract_af3, rtn=True, norm=norm)
        plt.xlabel('AF field [mT]')
        plt.grid()
        plt.show()