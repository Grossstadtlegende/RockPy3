__author__ = 'volk'
import matplotlib.pyplot as plt
import numpy as np
import time
from copy import deepcopy

from RockPy.core import measurement
from RockPy.core.data import RockPyData
import RockPy.Visualize.Features.rmp
import RockPy.Visualize.core
from RockPy.utils.general import to_list

class ThermoCurve(measurement.Measurement):
    @classmethod
    def measurement_result(cls, sample_obj, result, result_mtype=None,
                           m_idx=0,
                           calculation_parameter=None,
                           color=None, linestyle='-', marker='.', label='',  # todo change to dafult None
                           **parameter):
        """
        Parameter
        ---------
            sample_obj: RockPy.Sample
                The sample object the measurement should be added to
            result: str
                the result method to be called and used as y-data
            mtype: str
                default = None
                the mtype the result is calculated from.
                if None all possible mtypes are used
                else only the ones specified are used
            m_idx: int
            calculation_parameter: dict
                dictionary with the calculation parameter
            color: str
                color for plotting
            linestyle: str
                linestyle for plotting
            marker: str
                marker for plotting
            label : str
                label for the legend of the plot

        return measurement_result instance of measurement depending on parameters


        """

        if not calculation_parameter:
            calculation_parameter = {}
        # initialize measuremnt lists for up_temp / down_temp
        utm, dtm = [], []

        m_with_result = [m for m in sample_obj.measurements if m.has_result(result)]
        cls.logger.debug(
            'FOUND %i [of %i] measurements with result %s' % (len(m_with_result), len(sample_obj.measurements), result))

        m_with_series = [m for m in m_with_result if m.has_series('temp')]
        cls.logger.debug(
            'FOUND %i [of %i] measurements with series %s' % (len(m_with_series), len(m_with_result), 'temp'))

        # filter for mtype if given
        if result_mtype:
            # check for abbreviations and correct
            result_mtype = to_list(result_mtype)
            result_mtype = [RockPy.mtype_ftype_abbreviations_inversed[i]
                            if i in RockPy.mtype_ftype_abbreviations_inversed
                            else i
                            for i in result_mtype]

            m_with_series = [m for m in m_with_series if m.mtype in result_mtype]
            cls.logger.debug(
                'FOUND %i [of %i] measurements with mtype %s' % (len(m_with_series), len(m_with_series), result_mtype))

        if any(m.has_series('up_temp') for m in m_with_series):
            utm = [m for m in sample_obj.measurements if m.has_series('up_temp')]
            cls.logger.debug('FOUND %i [of %i] measurements with series %s' % (
                len(m_with_result), len(sample_obj.measurements), 'up_temp'))
        if any(m.has_series('down_temp') for m in m_with_series):
            dtm = [m for m in sample_obj.measurements if m.has_series('up_temp')]
            cls.logger.debug('FOUND %i [of %i] measurements with series %s' % (
                len(m_with_result), len(sample_obj.measurements), 'down_temp'))
        else:
            cls.logger.warning('NO heating/cooling information found assuming heating (up_temp)')
            utm = m_with_series

        mdata = {'up_temp': [[m.get_sval('temp'),
                              getattr(m, 'result_' + result)(**calculation_parameter)[0],
                              getattr(m, 'result_' + result)(**calculation_parameter)[1],
                              time.clock()]
                             for m in utm],
                 'down_temp': [[m.get_sval('temp'),
                                getattr(m, 'result_' + result)(**calculation_parameter)[0],
                                getattr(m, 'result_' + result)(**calculation_parameter)[1],
                                time.clock()]
                               for m in dtm]}

        mdata = {k: np.array(v) for k, v in mdata.iteritems()}

        for k in mdata.keys():
            try:
                mdata[k] = RockPyData(column_names=['temp', result, 'std', 'time'], data=np.array(mdata[k]))
                mdata[k].define_alias('mag', result)
            except TypeError:
                mdata[k] = None
        return cls(sample_obj=sample_obj, mtype='thermocurve', mdata=deepcopy(mdata),
                   fpath=None, ftype='result',
                   color=color, **parameter)

    # def __init__(self, sample_obj,
    #              mtype, fpath, ftype,
    #              **options):
    #     self._data = {'up_temp': None,
    #                   'down_temp': None}
    #
    #     super(ThermoCurve, self).__init__(sample_obj, mtype, fpath, ftype)

    def format_vftb(self):
        data = self.ftype_data.out_thermocurve()
        header = self.ftype_data.header
        if len(data) > 2:
            print('LENGTH of ftype.out_thermocurve =! 2. Assuming data[0] = heating data[1] = cooling')
            # self.log.warning('LENGTH of machine.out_thermocurve =! 2. Assuming data[0] = heating data[1] = cooling')
        if len(data) > 1:
            self._raw_data['up_temp'] = RockPyData(column_names=header, data=data[0])
            self._raw_data['down_temp'] = RockPyData(column_names=header, data=data[1])
        else:
            print('LENGTH of ftype.out_thermocurve < 2.')
            # self.log.error('LENGTH of machine.out_thermocurve < 2.')

    def format_vsm(self):
        data = self.ftype_data.out
        header = self.ftype_data.header
        segments = self.ftype_data.segment_info
        aux = np.array([j for i in data for j in i])  # combine all data arrays
        a = np.array([(i, v) for i, v in enumerate(np.diff(aux, axis=0)[:, 0])])

        sign = np.sign(np.diff(aux, axis=0)[:, 1])

        threshold = 3
        zero_crossings = [i + 1 for i in xrange(len(a[:, 1]) - 1)
                          if a[i, 1] > 0 > a[i + 1, 1] and a[i, 1] > 0 > a[i + 2, 1]
                          or a[i, 1] < 0 < a[i + 1, 1] and a[i, 1] < 0 < a[i + 2, 1]]
        zero_crossings = [0] + zero_crossings  # start with zero index
        zero_crossings += [len(aux)]  # append last index

        ut = 0  # running number warming
        dt = 0  # running number cooling

        for i, v in enumerate(zero_crossings):
            if v < zero_crossings[-1]:  # prevents index Error
                if sum(a[v:zero_crossings[i + 1], 1]) < 0:  # cooling
                    name = 'cool%02i' % (ut)
                    ut += 1
                else:
                    name = 'warm%02i' % (dt)
                    dt += 1
                data = aux[v:zero_crossings[i + 1] + 1]
                rpd = RockPyData(column_names=header, data=data)
                rpd.rename_column('temperature', 'temp')
                rpd.rename_column('moment', 'mag')
                self._data.update({name: rpd})

    def delete_segment(self, segment):
        self._data.pop(segment)

    @property
    def ut(self):
        """
        returns a RPdata with all warming data
        """
        out = None
        for i in self.data:
            if 'warm' in i:
                if not out:
                    out = self.data[i]
                else:
                    out = out.append_rows(self.data[i])
        return out

    @property
    def dt(self):
        """
        returns a RPdata with all cooling data
        """
        out = None
        for i in self.data:
            if 'cool' in i:
                if not out:
                    out = self._data[i]
                else:
                    out = out.append_rows(self.data[i])
        return out

    def plt_thermocurve(self):
        fig = RockPy.Visualize.core.generate_plots(n=1)
        ax = RockPy.Visualize.core.get_subplot(fig, 0)
        ax2 = ax.twinx()
        RockPy.Visualize.Features.rmp.mom_temp(ax, mobj=self, plt_opt=dict(marker=''))
        RockPy.Visualize.Features.rmp.dmom_dtemp(ax2, mobj=self, plt_opt=dict(marker=''))
        ax.grid()
        plt.show()
