__author__ = 'volk'
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from RockPy.core.data import RockPyData
from RockPy.core import measurement
import RockPy
from RockPy.utils import general
from RockPy.utils import ignored
import RockPy.Visualize.core
from RockPy.Visualize.Features import arai as arai_feature
from RockPy.Visualize.Features import dunlop as dunlop_feature
import RockPy.core
from RockPy.core.measurement import calculate, result


class Thellier(measurement.Measurement):
    """
    Measurement class for Thellier Thellier Experiments

    **Results**

    Arai plot statistics:
       result_n:

       result_slope:

       result_sigma:

       result_x_int:

       result_y_int:

       result_b_anc:

       result_sigma_b_anc:

       result_f:

       result_f_vds:

       result_frac:

       result_beta:

       result_g:

       result_gap:

       result_q:

       result_w:

    Directional statistics

    pTRM check statistics

    pTRM tails check statistics

    Additivity check statistics


    """
    # todo format_sushibar
    # todo format_jr6

    _standard_parameter = {'slope': {'t_min': 20, 't_max': 700, 'component': 'mag', 'b_lab': 35.0},
                           'b_anc': {'b_lab': 35.0}}

    _standard_simulation = dict(b_lab=35.0, b_anc=50.0)

    @classmethod
    def simulate(cls, sample_obj, m_idx=None, **parameter):
        """
        return simulated instance of measurement depending on parameters
        """
        Thellier._standard_simulation.update(parameter)
        parameter = Thellier._standard_simulation

        b_lab = parameter.get('b_lab', 35.0)

        aniso = parameter.get('aniso', [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        noise = parameter.get('noise', 1.e-3)
        check_freq = parameter.get('check_freq', 2)
        # temps = parameter.get('temps', [20, 300, 450, 490, 500, 510, 515, 520, 525, 530, 535, 540, 545, 550, 560])
        temps = parameter.get('temps', [20] + np.arange(100, 650, 25.))
        max_moment = parameter.get('max_moment', 1.0)

        th_steps = []
        pt_steps = []

        ac_steps = []
        ck_steps = []
        tr_steps = []

        # checks
        n = 0
        for i, v in enumerate(temps):
            th_steps.append([n, v])
            n += 1
            if (i - 1) % check_freq == 0 and len(temps) - 2 >= i >= check_freq:
                ck_steps.append([n, temps[i - check_freq], temps[i]])
                n += 1
            pt_steps.append([n, v])
            n += 1
            if (i - 1) % check_freq == 0 and len(temps) - 2 >= i >= check_freq:
                ac_steps.append([n, temps[i - check_freq], temps[i]])
                n += 1
            if (i - 1) % check_freq == 0 and len(temps) - 2 >= i >= check_freq:
                tr_steps.append([n, temps[i]])
                n += 1

        th_steps = np.array(th_steps)
        pt_steps = np.array(pt_steps)  # [1:]
        ac_steps = np.array(ac_steps)
        ck_steps = np.array(ck_steps)
        tr_steps = np.array(tr_steps)

        """ Generate data """

        mdata = {'th': None, 'pt': None}  # , 'ac': None, 'ck': None, 'tr': None}  #initialize

        th_data = np.linspace(max_moment, 0, len(temps)).T

        mdata['th'] = RockPyData(column_names=['temp', 'x', 'y', 'z', 'sm', 'time'])
        mdata['pt'] = RockPyData(column_names=['temp', 'x', 'y', 'z', 'sm', 'time'])
        mdata['ac'] = RockPyData(column_names=['temp', 'x', 'y', 'z', 'sm', 'time'])
        mdata['ck'] = RockPyData(column_names=['temp', 'x', 'y', 'z', 'sm', 'time'])
        mdata['tr'] = RockPyData(column_names=['temp', 'x', 'y', 'z', 'sm', 'time'])

        mdata['th']['temp'] = th_steps[:, 1]
        mdata['th']['time'] = th_steps[:, 0]
        mdata['th']['x'] = th_data
        mdata['th']['y'] = th_data
        mdata['th']['z'] = th_data

        mdata['pt']['temp'] = pt_steps[:, 1]
        mdata['pt']['time'] = pt_steps[:, 0]
        mdata['pt']['x'] = np.ones(len(pt_steps[:, 0])) * max_moment
        mdata['pt']['y'] = np.ones(len(pt_steps[:, 0])) * max_moment
        mdata['pt']['z'] = np.ones(len(pt_steps[:, 0])) * max_moment

        mdata['nrm'] = mdata['th'].filter_idx([0])

        ck_index = [i for i, v in enumerate(mdata['pt']['temp'].v) if v in ck_steps[:, 1]]
        ck_th_index = [i for i, v in enumerate(mdata['th']['temp'].v) if v in ck_steps[:, 2]]
        ck_thj_index = [i for i, v in enumerate(mdata['th']['temp'].v) if v in ck_steps[:, 1]]

        pt_copy = mdata['pt'].filter_idx(ck_index)
        thi_copy = mdata['th'].filter_idx(ck_th_index)
        thj_copy = mdata['th'].filter_idx(ck_thj_index)

        mdata['ck'] = thi_copy
        mdata['ck']['x'] = thi_copy['x'].v + (pt_copy['x'].v - thj_copy['x'].v)
        mdata['ck']['y'] = thi_copy['y'].v + (pt_copy['y'].v - thj_copy['y'].v)
        mdata['ck']['z'] = thi_copy['z'].v + (pt_copy['z'].v - thj_copy['z'].v)
        mdata['ck']['time'] = ck_steps[:, 0]
        mdata['ck']['temp'] = ck_steps[:, 1]

        tr_index = [i for i, v in enumerate(mdata['th']['temp'].v) if v in tr_steps[:, 1]]
        mdata['tr'] = mdata['th'].filter_idx(tr_index)
        mdata['tr']['time'] = tr_steps[:, 0]

        ac_pt_index = [i for i, v in enumerate(mdata['pt']['temp'].v) if v in ac_steps[:, 2]]
        ac_th_index = [i for i, v in enumerate(mdata['th']['temp'].v) if v in ac_steps[:, 1]]

        pt_copy = mdata['pt'].filter_idx(ac_pt_index)
        thi_copy = mdata['th'].filter_idx(ac_th_index)

        mdata['ac'] = pt_copy
        mdata['ac']['x'] = (pt_copy['x'].v - (max_moment - thi_copy['x'].v))
        mdata['ac']['y'] = (pt_copy['y'].v - (max_moment - thi_copy['y'].v))
        mdata['ac']['z'] = (pt_copy['z'].v - (max_moment - thi_copy['z'].v))
        mdata['ac']['time'] = ac_steps[:, 0]
        mdata['ac']['temp'] = ac_steps[:, 1]

        for dtype in mdata:
            if noise:
                for d in ['x', 'y', 'z']:
                    mdata[dtype][d] = mdata[dtype][d].v + np.random.normal(0.0, noise, len(mdata[dtype][d].v))
                    # mdata[dtype][d] = np.round(mdata[dtype][d].v)
            mdata[dtype].define_alias('m', ('x', 'y', 'z'))
            mdata[dtype] = mdata[dtype].append_columns('mag', mdata[dtype].magnitude('m'))

        return cls(sample_obj, mtype='thellier', fpath=None, mdata=mdata, ftype='simulation', **parameter)

    def __init__(self, sample_obj,
                 mtype, fpath, ftype,
                 **options):

        # # ## initialize data
        self.steps = ['nrm', 'th', 'pt', 'ac', 'tr', 'ck', 'ptrm', 'sum', 'difference']
        # self._data = {}

        super(Thellier, self).__init__(sample_obj, mtype, fpath, ftype, **options)
        self.reset__data()

    def reset__data(self, recalc_mag=True):
        self._data.update({'ptrm': self._ptrm(recalc_mag)})
        self._data.update({'sum': self._sum(recalc_mag)})
        self._data.update({'difference': self._difference(recalc_mag)})

    @property
    def data(self):
        if not 'ptrm' in self._data.keys():
            self.reset__data()
        return self._data

    @data.setter
    def data(self, data_dict):
        for dtype in data_dict:
            setattr(self, dtype, data_dict[dtype])
        if any(i not in data_dict for i in ['ptrm', 'sum', 'difference']):
            self.reset__data()

    def format_cryomag(self):
        """
        Formats cryomag output dictionary into thellier measurement data format.

        Beware: NRM step has to be called NRM or TH

        :return:
        """
        steps = self.ftype_data.steps
        data = self.ftype_data.get_float_data()
        row_labels = [v + '[%.0f]' % (data[i, 0]) for i, v in enumerate(steps)]

        self.all_data = RockPyData(column_names=self.ftype_data.float_header,
                                   data=data,
                                   row_names=row_labels)

        self.all_data.rename_column('step', 'temp')
        self.all_data = self.all_data.append_columns('time', self.ftype_data.get_time_data())
        nrm_idx = [i for i, v in enumerate(steps) if v == 'nrm']
        self.ftype_data.get_time_data()
        # generating the palint data for all steps
        for step in ['nrm', 'th', 'pt', 'ac', 'tr', 'ck']:
            idx = [i for i, v in enumerate(steps) if v == step]
            if step in ['th', 'pt']:
                idx.append(nrm_idx[0])
            if step == 'nrm' and len(idx) == 0:
                idx = [i for i, v in enumerate(steps) if v == 'th'][0]
            if len(idx) != 0:
                rp_data = self.all_data.filter_idx(idx)  # finding step_idx
                rp_data = rp_data.eliminate_duplicate_variable_rows(substfunc='last')
                rp_data.define_alias('m', ('x', 'y', 'z'))
                rp_data = rp_data.append_columns('mag', rp_data.magnitude('m'))
                rp_data = rp_data.sort('temp')
                rp_data.define_alias('variable', 'temp')
                self._data.update({step: rp_data})
            else:
                self._data.update({step: None})
        self.reset__data()

    def format_tdt(self):
        """
        formats the data (a dictionary) from the machine readin into RockPyData objects
        :return:
        """
        # get header, units and data from machine
        header = self.ftype_data.header

        if self.sample_obj.volume:
            data = self.ftype_data.data('unnormalized_data', volume=self.sample_obj.volume)
            units = ['C', 'A m2', 'A m2', 'A m2', 's']
        else:
            data = self.ftype_data.data('formatted_data')
            units = ['C', 'A/m', 'A/m', 'A/m', 's']

        # store data in RockPyData instances in the _data dictionary
        for dtype in data.keys():
            rp_data = RockPyData(column_names=header, units=units, data=data[dtype])
            rp_data = rp_data.eliminate_duplicate_variable_rows(substfunc='last')
            rp_data.define_alias('m', ('x', 'y', 'z'))
            rp_data = rp_data.append_columns('mag', rp_data.magnitude('m'))

            rp_data = rp_data.sort('temp')
            rp_data.define_alias('variable', 'temp')
            self._data.setdefault(dtype, rp_data)
            # self.data[dtype]._row_names = ['%s[%i]'%(dtype, i) for i in self.data[dtype]['temp'].v]

        self.reset__data()

    def _ptrm(self, recalc_mag=True):
        idx = self._get_idx_equal_val('pt', 'th')

        pt = self._data['pt'].filter_idx(idx[:, 0])
        th = self._data['th'].filter_idx(idx[:, 1])

        ptrm = pt - th
        row_labels = ['ptrm[%i]' % i for i in pt['temp'].v]

        ptrm['time'] = pt['time'].v  # copy old pt times into ptrm
        if recalc_mag:
            ptrm.define_alias('m', ('x', 'y', 'z'))
            if not 'mag' in ptrm.column_names:
                ptrm = ptrm.append_columns('mag', ptrm.magnitude('m'))
            ptrm['mag'] = ptrm.magnitude('m')
        ptrm._row_names = row_labels
        return ptrm

    def _sum(self, recalc_mag=True):
        """
        Calculates the sum of the pt and th steps, for equal temperature steps

        Parameters
        ----------
           recalc_mag: Bool
              if True, 'mag' will be re-calculated. Not always a good idea.
        """
        idx = self._get_idx_equal_val('pt', 'th')
        pt = self._data['pt'].filter_idx(idx[:, 0])
        th = self._data['th'].filter_idx(idx[:, 1])
        pt_th_sum = th + pt - th
        if recalc_mag:
            pt_th_sum.define_alias('m', ('x', 'y', 'z'))
            if not 'mag' in pt_th_sum.column_names:
                pt_th_sum = pt_th_sum.append_columns('mag', pt_th_sum.magnitude('m'))
            pt_th_sum['mag'] = pt_th_sum.magnitude('m')
        return pt_th_sum

    def _difference(self, recalc_mag=True):
        idx = self._get_idx_equal_val('pt', 'th')
        pt = self._data['pt'].filter_idx(idx[:, 0])
        th = self._data['th'].filter_idx(idx[:, 1])
        ptrm = pt - th
        difference = th - ptrm
        if recalc_mag:
            difference.define_alias('m', ('x', 'y', 'z'))
            if not 'mag' in difference.column_names:
                difference = ptrm.append_columns('mag', difference.magnitude('m'))
            else:
                difference['mag'] = difference.magnitude('m')
        return difference

    def _get_idx_tmin_tmax(self, step, t_min, t_max):
        idx = (getattr(self, step)['temp'].v <= t_max) & (t_min <= getattr(self, step)['temp'].v)
        return idx

    def _get_idx_equal_val(self, step_x, step_y, key='temp'):

        idx = np.array([(xi, yi) for xi, v1 in enumerate(self._data[step_x][key].v)
                        for yi, v2 in enumerate(self._data[step_y][key].v)
                        if v1 == v2])
        return idx

    def _get_idx_step_var_val(self, step, var, val, *args):
        """
        returns the index of the closest value with the variable(var) and the step(step) to the value(val)

        option: inverse:
           returns all indices except this one

        """
        out = [np.argmin(abs(self.data[step][var].v - val))]
        return out

    def correct_step(self, step='th', var='variable', val='last', initial_state=True):
        """
        corrects the remaining moment from the last th_step

        :param step:
        :param var:
        :param val:
        :param initial_state: also corrects the iinitial state if one exists
        """

        try:
            calc_data = self.data[step]
        except KeyError:
            self.log.error('REFERENCE << %s >> can not be found ' % (step))

        if val == 'last':
            val = calc_data[var].v[-1]
        if val == 'first':
            val = calc_data[var].v[0]

        idx = self._get_idx_step_var_val(step=step, var=var, val=val)

        correction = self.th.filter_idx(idx)  # correction step
        for dtype in ['nrm', 'th', 'pt', 'ac', 'ck', 'tr']:
            # store variables so calculation does not affect
            # vars = self.data[dtype]['temp'].v
            # calculate correction
            self._data[dtype]['m'] = self._data[dtype]['m'].v - correction['m'].v
            # refill variables with original data
            # self.data[dtype]['temp'] = vars
            # recalc mag for safety
            self.data[dtype]['mag'] = self.data[dtype].magnitude(('x', 'y', 'z'))
        self.reset__data()

        if self.initial_state and initial_state:
            for dtype in self.initial_state.data:
                self.initial_state.data[dtype]['m'] = self.initial_state.data[dtype]['m'].v - correction['m'].v
                self.initial_state.data[dtype]['mag'] = self.initial_state.data[dtype].magnitude(('x', 'y', 'z'))
        return self

    # ## plotting functions
    def plt_dunlop(self):
        plt.plot(self.data['th']['temp'].v, self.data['th']['mag'].v, '.-', zorder=1)
        # plt.plot(self.ptrm['temp'], self.ptrm['moment'], '.-', zorder=1)
        plt.plot(self.data['ptrm']['temp'].v, self.data['ptrm']['mag'].v, '.-', zorder=1)
        plt.plot(self.data['sum']['temp'].v, self.data['sum']['mag'].v, '.--', zorder=1)
        plt.plot(self.data['tr']['temp'].v, self.data['tr']['mag'].v, 's')
        plt.grid()
        plt.title('Dunlop Plot %s' % (self.sample_obj.name))
        plt.xlabel('Temperature [%s]' % ('C'))
        plt.ylabel('Moment [%s]' % ('Am2'))
        # plt.xlim([min(self.th['temp'].v), max(self.th['temp']).v])
        plt.show()

    def plt_arai(self, **options):
        equal = set(self.th['temp'].v) & set(self.ptrm['temp'].v)
        max_mag = max(set(self.th['mag'].v) | set(self.ptrm['mag'].v))
        idx = [i for i, v in enumerate(self.th['temp'].v) if v in equal]
        th = self.th.filter_idx(idx)
        plt.plot(self.ptrm['mag'].v, th['mag'].v, 'o', zorder=1, markersize=5)
        plt.plot([min(self.ptrm['mag'].v), max(self.ptrm['mag'].v)],
                 self.result_slope().v * np.array(
                     [min(self.ptrm['mag'].v), max(self.ptrm['mag'].v)]) + self.result_y_int().v, '--')
        plt.grid()
        plt.title('Arai Diagram %s' % (self.sample_obj.name))
        plt.xlabel('NRM remaining [%s]' % ('C'))
        plt.ylabel('pTRM gained [%s]' % ('Am2'))
        plt.ylim([0, max_mag * 1.1])
        plt.show()

    def plt_thellier(self):
        fig = RockPy.Visualize.core.generate_plots(2)
        for i, feature in enumerate(['arai', 'dunlop']):
            ax = RockPy.Visualize.core.get_subplot(fig, i)
            if feature == 'arai':
                arai_feature.arai_points(ax, self)
                arai_feature.arai_fit(ax, self)
                arai_feature.add_ck_check(ax, self)
                ax.grid()
                ax.set_title('Arai Diagram %s' % (self.sample_obj.name))
                ax.set_xlabel('NRM remaining [%s]' % ('Am2'))
                ax.set_ylabel('pTRM gained [%s]' % ('Am2'))
                ax.set_ylim([0, max(self.th['mag'].v) * 1.1])
                ax.set_xlim([0, max(self.ptrm['mag'].v) * 1.1])

            if feature == 'dunlop':
                max_mag = max(set(self.th['mag'].v) | set(self.ptrm['mag'].v))
                dunlop_feature.dunlop_data(ax, self)
                ax.set_title('Dunlop Plot %s' % (self.sample_obj.name))
                ax.set_xlabel('Temperature [%s]' % ('C'))
                ax.set_ylabel('magnetic moment [%s]' % ('Am2'))
                ax.grid()
                ax.legend(loc='best', )
                ax.set_ylim([0, max_mag * 1.1])

        plt.tight_layout()
        plt.show()

    def delete_temp(self, temp):
        for step in self.steps:
            o_len = len(getattr(self, step)['temp'].v)
            idx = [i for i, v in enumerate(getattr(self, step)['temp'].v) if v != temp]
            if o_len - len(idx) != 0:
                self.log.info(
                    'DELETING << %i, %s >> entries for << %.2f >> temperature' % (o_len - len(idx), step, temp))
                setattr(self, step, getattr(self, step).filter_idx(idx))
            else:
                self.log.debug('UNABLE to find entriy for << %s, %.2f >> temperature' % (step, temp))

    """ RESULT SECTION """

    """
    Arai plot statistics
    ====================
    
    A note on data indexing
    +++++++++++++++++++++++
        
    Statistic: :math:`i` and :math:`n_{max}`
    
    The index :math:`i` is used to denote the :math:`i^{th}` temperature step of the paleointensity experiment.
    :math:`i` is used to index Arai plot data (e.g., :math:`x_i`, or :math:`y_i`) and ranges from :math:`i=1` to
    :math:`n_{max}`, where :math:`n_{max}` is the total number of steps on the Arai plot.
    
     
    
    Statistic: :math:`start` and :math:`end`
    
    :math:`start` and :math:`end` denote the :math:`i` indices of the selected steps used for analyzing the
    paleointensity results. :math:`i=start` denotes the first selected data point and :math:`i=end` denotes the last.
    
     
    
    Statistic: :math:`T_{min}` and :math:`T_{max}`
    
    The minimum and maximum temperatures used for the best-fit linear segment on the Arai plot, 
    where :math:`T_{min} \equiv T_{i=start}` and :math:`T_{max} \equiv T_{i=end}`.
    
    """
    ####################################################################################################################
    """ RESULTS CALCULATED USING CALCULATE_SLOPE  METHODS """

    @calculate
    def calculate_slope(self, t_min=20, t_max=700, component='mag',
                        **non_calculation_parameters):
        """
        calculates the least squares slope for the specified temperature interval

        :param parameter:

        """

        # get equal temperature steps for both th and ptrm measurements
        equal_steps = list(set(self.th['temp'].v) & set(self.ptrm['temp'].v))

        # Filter data for the equal steps and filter steps outside of tmin-tmax range
        # True if step between t_min, t_max
        th_steps = (t_min <= self.th['temp'].v) & (self.th['temp'].v <= t_max)
        ptrm_steps = (t_min <= self.ptrm['temp'].v) & (self.ptrm['temp'].v <= t_max)

        th_data = self.th.filter(th_steps)  # filtered data for t_min t_max
        ptrm_data = self.ptrm.filter(ptrm_steps)  # filtered data for t_min t_max

        # filtering for equal variables
        th_idx = [i for i, v in enumerate(th_data['temp'].v) if v in equal_steps]
        ptrm_idx = [i for i, v in enumerate(ptrm_data['temp'].v) if v in equal_steps]

        th_data = th_data.filter_idx(th_idx)  # filtered data for equal t(th) & t(ptrm)
        ptrm_data = ptrm_data.filter_idx(ptrm_idx)  # filtered data for equal t(th) & t(ptrm)

        data = RockPyData(['th', 'ptrm'])

        # setting the data
        data['th'] = th_data[component].v
        data['ptrm'] = ptrm_data[component].v

        slope, sigma, y_int, x_int = data.lin_regress('ptrm', 'th')

        self.results['slope'] = [[[slope, sigma]]]
        self.results['sigma'] = sigma
        self.results['y_int'] = y_int
        self.results['x_int'] = x_int
        self.results['n'] = len(th_data[component].v)

    @result
    def result_slope(self, t_min=20, t_max=700, component='mag',
                     recalc=False,
                     **non_calculation_parameters):
        """
        Gives result for calculate_slope(t_min, t_max), returns slope value if not calculated already
        """
        pass

    @result
    def result_n(self, t_min=20, t_max=700, component='mag',
                 recalc=False,
                 calculation_method='slope',
                 **non_calculation_parameters):
        """
        Number of steps used for the calculation of the best fit line
        """
        pass

    @result
    def result_sigma(self, t_min=20, t_max=700, component='mag',
                     recalc=False,
                     calculation_method='slope',
                     **non_calculation_parameters):
        """
        Standard deviation of the best fit line
        """
        pass

    @result
    def result_x_int(self, t_min=20, t_max=700, component='mag',
                     recalc=False,
                     calculation_method='slope',
                     **non_calculation_parameters):
        """
        """  # todo write doc
        pass

    @result
    def result_y_int(self, t_min=20, t_max=700, component='mag',
                     recalc=False,
                     calculation_method='slope',
                     **non_calculation_parameters):
        """
        """  # todo write doc
        pass

    ####################################################################################################################
    ''' NOT ONLY SLOPE BASED '''

    @calculate  # todo maybe some results should be property, because they rely on another result heavily
    def calculate_b_anc(self, t_min=20, t_max=700, component='mag', b_lab=35.0,
                        **non_calculation_parameters):
        """
        calculates the :math:`B_{anc}` value for a given lab field in the specified temperature interval.

        :param parameter:

        Note
        ----
            This calculation method calls calculate_slope if you call it again afterwards, with different
            calculation_parameters, it will not influence this result. Therfore you have to be careful when calling
            this.
        """
        b_anc = self.result_slope(t_min=t_min, t_max=t_max, component=component, **non_calculation_parameters)
        self.results['b_anc'] = [[[abs(b_lab * b_anc[0]), abs(b_lab * b_anc[1])]]]
        self.results['sigma_b_anc'] = abs(b_lab * b_anc[1])

    @result
    def result_b_anc(self, t_min=20, t_max=700, component='mag', b_lab=35.0,
                     recalc=False,
                     **non_calculation_parameters):  # todo write comment
        pass

    @result
    def result_sigma_b_anc(self, t_min=20, t_max=700, component='mag', b_lab=35.0,
                           recalc=False, calculation_method='b_anc',
                           **non_calculation_parameters):  # todo write comment
        pass

    ####################################################################################################################
    """ F """

    @calculate
    def calculate_f(self, **parameter):
        """

        The remanence fraction, f, was defined by Coe et al. (1978) as:

        .. math::

           f =  \\frac{\\Delta y^T}{y_0}

        where :math:`\Delta y^T` is the length of the NRM/TRM segment used in the slope calculation.


        :param parameter:
        :return:

        """
        delta_y_dash = self.delta_y_dash(**parameter)
        y_int = self.results['y_int'].v
        self.results['f'] = delta_y_dash / abs(y_int)

    @result
    def result_f(self, t_min=20, t_max=700, recalc=False, **options):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ F_VDS """

    @calculate
    def calculate_f_vds(self, t_min=20, t_max=700,
                        **non_calculation_parameters):
        """

        NRM fraction used for the best-fit on an Arai diagram calculated as a vector difference sum (Tauxe and Staudigel, 2004).

        .. math::

           f_{VDS}=\\frac{\Delta{y'}}{VDS}

        :param parameter:
        :return:

        """
        delta_y = self.delta_y_dash(t_min=t_min, t_max=t_max, **non_calculation_parameters)
        VDS = self.calculate_vds()
        self.results['f_vds'] = delta_y / VDS

    @result
    def result_f_vds(self, t_min=20, t_max=700, recalc=False, **non_calculation_parameters):
        pass

    ####################################################################################################################
    """ FRAC """

    @calculate
    def calculate_frac(self, t_min=20, t_max=700, **non_calculation_parameters):
        """

        NRM fraction used for the best-fit on an Arai diagram determined entirely by vector difference sum
        calculation (Shaar and Tauxe, 2013).

        .. math::

            FRAC=\\frac{\sum\limits_{i=start}^{end-1}{ \left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right| }}{VDS}

        :param parameter:
        :return:

        """

        NRM_sum = np.sum(np.fabs(self.calculate_vd(t_min=t_min, t_max=t_max, **non_calculation_parameters)))
        VDS = self.calculate_vds()
        self.results['frac'] = NRM_sum / VDS

    @result
    def result_frac(self, t_min=20, t_max=700, recalc=False, **options):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ BETA """

    @calculate
    def calculate_beta(self, t_min=20, t_max=700, component='mag',
                       **non_calculation_parameters):
        """

        :math:`\beta` is a measure of the relative data scatter around the best-fit line and is the ratio of the
        standard error of the slope to the absolute value of the slope (Coe et al., 1978)

        .. math::

           \\beta = \\frac{\sigma_b}{|b|}


        :param parameters:
        :return:

        """

        slope = self.result_slope(t_min=t_min, t_max=t_max, component=component, **non_calculation_parameters)[0]
        sigma = self.result_sigma(t_min=t_min, t_max=t_max, component=component, **non_calculation_parameters)[0]
        self.results['beta'] = sigma / abs(slope)

    @result
    def result_beta(self, t_min=20, t_max=700, recalc=False, **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ G """

    @calculate
    def calculate_g(self, t_min=20, t_max=700, component='mag',
                    **non_calculation_parameters):
        """

        Gap factor: A measure of the gap between the points in the chosen segment of the Arai plot and the least-squares
        line. :math:`g` approaches :math:`(n-2)/(n-1)` (close to unity) as the points are evenly distributed.

        """
        y_dash = self.y_dash(t_min=t_min, t_max=t_max, component=component, **non_calculation_parameters)
        delta_y_dash = self.delta_y_dash(t_min=t_min, t_max=t_max, component=component, **non_calculation_parameters)
        y_dash_diff = [(y_dash[i + 1] - y_dash[i]) ** 2 for i in range(len(y_dash) - 1)]
        y_sum_dash_diff_sq = np.sum(y_dash_diff, axis=0)

        self.results['g'] = 1 - y_sum_dash_diff_sq / delta_y_dash ** 2

    @result
    def result_g(self, t_min=20, t_max=700, component='mag', recalc=False, **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ GAP MAX """

    def calculate_gap_max(self, t_min=20, t_max=700, **non_calculation_parameters):
        """
        The gap factor is a measure of the average Arai plot point spacing and may not represent extremes
        of spacing. To account for this Shaar and Tauxe (2013)) proposed :math:`GAP_{\text{MAX}}`, which is the maximum
        gap between two points determined by vector arithmetic.

        .. math::
           GAP_{\\text{MAX}}=\\frac{\\max{\{\\left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right|\}}_{i=start, \\ldots, end-1}}
           {\\sum\\limits_{i=start}^{end-1}{\\left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right|}}

        :return:

        """
        vd = self.calculate_vd(t_min=t_min, t_max=t_max)
        max_vd = np.max(vd)
        sum_vd = np.sum(vd)
        self.results['gap_max'] = max_vd / sum_vd

    @result
    def result_gap_max(self, t_min=20, t_max=700, recalc=False, **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ Q """

    @calculate
    def calculate_q(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """

        The quality factor (:math:`q`) is a measure of the overall quality of the paleointensity estimate and combines
        the relative scatter of the best-fit line, the NRM fraction and the gap factor (Coe et al., 1978).

        .. math::
           q=\\frac{\\left|b\\right|fg}{\\sigma_b}=\\frac{fg}{\\beta}

        :param parameter:
        :return:

        """
        self.logger.debug('CALCULATING\t quality parameter')

        beta = self.result_beta(t_min=t_min, t_max=t_max, component=component)[0]
        f = self.result_f(t_min=t_min, t_max=t_max, component=component)[0]
        gap = self.result_g(t_min=t_min, t_max=t_max, component=component)[0]
        self.results['q'] = (f * gap) / beta

    @result
    def result_q(self, t_min=20, t_max=700, component='mag', recalc=False, **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ W """

    @calculate
    def calculate_w(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
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
        q = self.result_q(t_min=t_min, t_max=t_max, component=component)[0]
        n = self.result_n(t_min=t_min, t_max=t_max, component=component)[0]
        self.results['w'] = q / np.sqrt((n - 2))

    @result
    def result_w(self, t_min=20, t_max=700, component='mag', recalc=False, **options):
        # todo write comment for this method
        pass

    """
    PTRM CHECK statistics
    =====================

    A pTRM check is a repeat TRM acquisition step to test for changes in a specimen's ability to acquire TRM at
    blocking temperatures below the temperature of the check. The difference between a pTRM check and the original TRM
    is calculated as the scalar intensity difference. That is,

    :math:
    dpTRMi,j = pTRM checki,j -TRMi = pTRM checki,j -xi,

    where pTRM checki,j is the pTRM check to the ith temperature step after heating to the jth tem- perature step.
    The order of the difference is such that pTRM checks smaller than the original TRM yield negative dpTRMi,j and pTRM
    checks larger than the original TRM give positive dpTRMi,j. For a pTRM check to be included in the analysis,
    both Ti and Tj must be less than or equal to Tmax.
    """


    ####################################################################################################################
    """ N_CK / N_PTRM """

    @calculate
    def calculate_n_ptrm(self, t_min=20, t_max=700, **non_calculation_parameters):
        """
        The number of pTRM checks (CK) used to analyze the best-fit segment on the Arai plot
        (i.e., the number of pTRMi,j with Ti <= Tmax and Tj <= Tmax).
        :param parameter:

        """
        # todo check if some of the ck have no equivalent th

        temps = self.ck['temp'].v

        # get maximum th/ptrm temperature, because ck temp has to be lower
        tt_temps = set(self.ptrm['temp'].v) & set(self.th['temp'].v)
        tt_max = max(i for i in tt_temps if i <= t_max)

        out = [i for i in temps
               if i >= t_min
               if i < tt_max]

        self.results['n_ptrm'] = len(out)

    @result
    def result_n_ptrm(self, t_min=20, t_max=700, recalc=False, **options):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ CK_CHECK_PERCENT """

    @calculate
    def calculate_ck_check_percent(self, t_min=20, t_max=700, component='mag', vectorial_diff=False,
                                   **non_calculation_parameters):
        """
        Maximum absolute difference produced by a pTRM check, normalized by the TRM acquired at that heating step.

        Parameters
        ----------
            t_min: float
            t_max: float
            component: str
            vectorial_diff: bool
                default:True
                if True the vectorial difference of dpTRM is calculated
        """

        dptrm = self.get_d_ptrm(t_min=t_min, t_max=t_max, vectorial_diff=vectorial_diff)

        # filter pTRM for equal temps
        ptrm = self.ptrm.filter([True if i in dptrm['temp'].v else False for i in self.ptrm['temp'].v])
        check_percent = max(np.fabs(dptrm[component].v) / ptrm[component].v * 100)
        self.results['ck_check_percent'] = abs(check_percent)

    @result
    def result_ck_check_percent(self, t_min=20, t_max=700, component='mag', vectorial_diff=False,
                                recalc=False, **options):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ DELTA_CK """

    @calculate
    def calculate_delta_ck(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        Maximum absolute difference produced by a pTRM check, normalized by the total TRM (obtained from the
        intersection of the best-fit line and the x-axis on an Arai plot; Leonhardt et al., 2004a).

        .. math::

           \delta{CK}=\\frac`{\max{ \left\{ \left| \delta{pTRM_{i,j}} \right| \right\} }_{i \leq end \textbf{ and } j
           \leq end}}{\left|X_{Int.}\right|}\times{100}
        """
        dptrm = self.get_d_ptrm(t_min=t_min, t_max=t_max)
        max_dptrm = np.max(np.fabs(dptrm[component].v))
        delta_ck = max_dptrm / abs(self.result_x_int(t_min=t_min, t_max=t_max, component=component)[0]) * 100
        self.results['delta_ck'] = delta_ck

    @result
    def result_delta_ck(self, t_min=20, t_max=700, component='mag', recalc=False, **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ DRAT """

    @calculate
    def calculate_drat(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        Maximum absolute difference produced by a pTRM check, normalized by the length of the best-fit line
        (Selkin and Tauxe, 2000).

        .. math::

           DRAT=\\frac`{\max{\left\{\left|\delta{pTRM_{i,j}} \right| \right\}}_{i \leq end \textbf{ and } j \leq end}}{L}\times{100},

        where `L` is the length of the best-fit line on the Arai plot. `L` is given by:

        .. math::

           L=\sqrt{ (\Delta{x'})^2 + (\Delta{y'})^2 }

        where `\Delta{x'}` and `\Delta{y'}` are TRM and NRM lengths of the best-fit line on the Arai plot, respectively (Section 3).
        """

        dptrm = self.get_d_ptrm(t_min=t_min, t_max=t_max)
        L = np.sqrt((self.delta_x_dash(t_min=t_min, t_max=t_max, component=component)) ** 2 +
                    (self.delta_y_dash(t_min=t_min, t_max=t_max, component=component)) ** 2)

        # get index of maximum value for the chosen component
        max_idx = np.argmax(abs(dptrm[component].v))
        drat = (abs(dptrm.filter_idx(max_idx)[component].v) / L) * 100
        self.results['drat'] = drat

    @result
    def result_drat(self, t_min=20, t_max=700, component='mag', recalc=False, **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ DRAT """

    @calculate
    def calculate_ck_max_dev(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        Maximum absolute difference produced by a pTRM check, normalized by the length of the TRM segment of the
        best-fit line on the Arai plot (Blanco et al., 2012).

        .. math::

           maxDEV=\\frac{ \max \\left\{ \\left| \delta pTRM_{i,j} \\right| \\right\}_{i \\leq end \\; \\textbf{and} \\; j \\leq end}  }{\Delta{x'}} \\times 100


        """
        dptrm = self.get_d_ptrm(t_min=t_min, t_max=t_max)

        max_idx = np.argmax(abs(dptrm[component].v))
        ck_max_dev = (dptrm.filter_idx(max_idx)[component].v /
                      self.delta_x_dash(t_min=t_min, t_max=t_max, component=component)) * 100
        self.results['ck_max_dev'] = abs(ck_max_dev)

    @result
    def result_ck_max_dev(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        # todo write comment for this method
        pass

    """
    Cumulative pTRM check parameters
    ********************************

    Most cumulative pTRM checks can be calculated in two fashions. The first method, is the summation of the signed
    pTRM differences (i.e., :math:`\pm \delta{pTRM}`), the second is to calculate the sum of the absolute pTRM difference
    (i.e., :math:`\left|\delta{pTRM} \right|`). The convention of the Standard Paleointensity Definition is to denote the
    second approach with a prime (`'`). For example, `CDRAT` is calculated by the first method and `CDRAT'`
    by the second.

    """

    ####################################################################################################################
    """ CDRAT """

    @calculate
    def calculate_cdrat(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        Cumulative `DRAT` (Kissel and Laj, 2004).

        .. math::

          CDRAT=\\frac{\\left|\\sum\\limits_{i=1}^{end}{\\delta{pTRM_{i,j}}}\\right|}{L}\\times{100} \\
          CDRAT'=\\frac{\\sum\\limits_{i=1}^{end}{\\left|\\delta{pTRM_{i,j}}\\right|}}{L}\\times{100}


        """

        dptrm = self.get_d_ptrm(t_min=t_min, t_max=t_max)

        L = np.sqrt((self.delta_x_dash(t_min=t_min, t_max=t_max, component=component)) ** 2 +
                    (self.delta_y_dash(t_min=t_min, t_max=t_max, component=component)) ** 2)

        signed_sum = np.abs(np.sum(dptrm[component].v))
        unsigned_sum = np.sum(np.fabs(dptrm[component].v))  # for cdrat'

        cdrat = (signed_sum / L) * 100
        cdrat_dash = (unsigned_sum / L) * 100

        self.results['cdrat'] = cdrat
        self.results['cdrat_dash'] = cdrat_dash

    @result
    def result_cdrat(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        # todo write comment for this method
        pass

    @result
    def result_cdrat_dash(self, t_min=20, t_max=700, component='mag',
                          calculation_method='cdrat',
                          **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ DRATS """

    @calculate
    def calculate_drats(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        Cumulative pTRM check difference normalized by the pTRM gained at the maximum temperature used for the
        best-fit on the Arai diagram (Tauxe and Staudigel, 2004).


        .. math::

           DRATS &=\\frac{\\left|\\sum\\limits_{i=1}^{end}{\\delta{pTRM_{i,j}}}\\right|}{x_{end}}\\times{100} \\\\
           DRATS' &=\\frac{\\sum\\limits_{i=1}^{end}{\\left|\\delta{pTRM_{i,j}}\\right|}}{x_{end}}\\times{100}
        """
        dptrm = self.get_d_ptrm(t_min=t_min, t_max=t_max)

        # finding last ptrm step for normalization of dck

        equal_steps = list(set(self.th['temp'].v) & set(self.ptrm['temp'].v))
        th_steps = (t_min <= self.th['temp'].v) & (self.th['temp'].v <= t_max)  # True if step between t_min, t_max
        ptrm_steps = (t_min <= self.ptrm['temp'].v) & (self.ptrm['temp'].v <= t_max)  # True if step betw. t_min, t_max

        ptrm_data = self.ptrm.filter(ptrm_steps)  # filtered data for t_min & t_max
        max_idx = np.argmax(ptrm_data['temp'].v)
        ptrm_data = ptrm_data.filter_idx(max_idx)

        signed_sum = np.abs(np.sum(dptrm[component].v))
        unsigned_sum = np.sum(np.fabs(dptrm[component].v))  # for cdrat' not used

        drats = (signed_sum / ptrm_data[component].v[0]) * 100
        drats_dash = (unsigned_sum / ptrm_data[component].v[0]) * 100
        self.results['drats'] = drats
        self.results['drats_dash'] = drats_dash

    @result
    def result_drats(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        # todo write comment for this method
        pass

    @result
    def result_drats_dash(self, t_min=20, t_max=700, component='mag',
                          calculation_method='drats',
                          **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ MEAN DRAT """

    @calculate
    def calculate_mean_drat(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        The average difference produced by a pTRM check, normalized by the length of the best-fit line.

        .. math::

           \\textrm{Mean }DRAT=\\frac{1}{n_{pTRM}}\\frac{\\left|\\sum\\limits_{i=1}^{end}{\\delta{pTRM_{i,j}}}\\right|}{L}\\times{100} \\
           \\textrm{Mean }DRAT'=\\frac{1}{n_{pTRM}}\\frac{\\sum\\limits_{i=1}^{end}{\\left|\\delta{pTRM_{i,j}}\\right|}}{L}\\times{100}
        """
        dptrm = self.get_d_ptrm(t_min=t_min, t_max=t_max)
        signed_sum = np.abs(np.sum(dptrm[component].v))
        unsigned_sum = np.sum(np.fabs(dptrm[component].v))  # for cdrat' not used

        mean_drat = (1 / self.result_n_ptrm(t_min=t_min, t_max=t_max, component=component)[0]) * \
                    (signed_sum / self.best_fit_line_length(t_min=t_min, t_max=t_max)) * 100
        mean_drat_dash = (1 / self.result_n_ptrm(t_min=t_min, t_max=t_max, component=component)[0]) * \
                         (unsigned_sum / self.best_fit_line_length(t_min=t_min, t_max=t_max)) * 100

        self.results['mean_drat'] = mean_drat
        self.results['mean_drat_dash'] = mean_drat_dash

    @result
    def result_mean_drat(self, t_min=20, t_max=700, component='mag', recalc=False, **non_calculation_parameters):
        # todo write comment for this method
        pass

    @result
    def result_mean_drat_dash(self, t_min=20, t_max=700, component='mag', recalc=False,
                              calculation_method='mean_drat',
                              **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ MEAN DEV """

    @calculate
    def calculate_mean_dev(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        Mean deviation of a pTRM check (Blanco et al., 2012).

        .. math::

           \\textrm{Mean }DEV=\\frac{1}{n_{pTRM}}\\frac{\\left|\\sum\\limits_{i=1}^{end}\\delta{pTRM_{i,j}}\\right|}{\\Delta{x'}}\\times{100} \\
           \\textrm{Mean }DEV'=\\frac{1}{n_{pTRM}}\\frac{\\sum\\limits_{i=1}^{end}\\left|\\delta{pTRM_{i,j}}\\right|}{\\Delta{x'}}\\times{100}

        """

        dptrm = self.get_d_ptrm(t_min=t_min, t_max=t_max)

        signed_sum = np.abs(np.sum(dptrm[component].v))
        unsigned_sum = np.sum(np.fabs(dptrm[component].v))  # for cdrat' not used

        mean_dev = (1 / self.result_n_ptrm(t_min=t_min, t_max=t_max, component=component)[0]) * \
                   (signed_sum / self.delta_x_dash(t_min=t_min, t_max=t_max, component=component)) * 100

        mean_dev_dash = (1 / self.result_n_ptrm(t_min=t_min, t_max=t_max, component=component)[0]) * \
                   (unsigned_sum / self.delta_x_dash(t_min=t_min, t_max=t_max, component=component)) * 100

        self.results['mean_dev'] = mean_dev
        self.results['mean_dev_dash'] = mean_dev_dash

    @result
    def result_mean_dev(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        # todo write comment for this method
        pass

    @result
    def result_mean_dev_dash(self, t_min=20, t_max=700, component='mag',
                             calculation_method='mean_dev',
                             **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ DELTA PAL """  # todo implement calculate_delta_pal

    @calculate
    def calculate_delta_pal(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        A measure of cumulative alteration determined by the difference of the alteration corrected intensity estimate
        (Valet et al., 1996) and the uncorrected estimate, normalized by the uncorrected estimate
        (Leonhardt et al., 2004a).

        We first calculate the cumulative sum of the pTRM checks up to the :math:`i^{th}` step of the experiment:

        .. math::

           \\mathbf{C}_i=\\sum\\limits_{l=1}^{l=i}{ \\mathbf{\\delta{pTRM}}_{l,j} }, ~~\\textrm{for} ~i=1,\\ldots, n_{max},

        where :math:`\mathbf{\delta{pTRM}}_{l,j}` is the vector difference between :math:`\mathbf{TRM}_l`
        and :math:`\mathbf{pTRM\_check}_{l,j}`, i.e.,

        .. math::

           \\mathbf{\\delta{pTRM}}_{l,j}=\\mathbf{TRM}_l - \\mathbf{pTRM\\_check}_{l,j}.

        When no pTRM check is performed :math:`\mathbf{\delta{pTRM}}_l=[0,0,0]`.

        The :math:`\mathbf{TRM}_i` vector is then corrected by adding the cumulative effect of the alteration,
        :math:`\mathbf{C}_i`:

        .. math::

           \mathbf{TRM}_i^*=\mathbf{TRM}_i+\mathbf{C}_{i}, ~~ \\textrm{for} ~i=1,\ldots, n_{max}.

        Since no pTRM check is performed at the first step:

        .. math::

           \mathbf{TRM}_1^*=\mathbf{TRM}_1.

        The corrected TRM values on the Arai plot (:math:`x_i^*`) can be calculated by determining the vector
        lengths of :math:`\mathbf{TRM}_i^*`.
        The corrected slope on the Arai plot (:math:`b^*`) can be calculated using the selected points and the
        standard approach outlined in Section 3.

        :math:`\delta{pal}` is then given by:

        .. math::

           \delta{pal}=\left|\\frac{b-b^*}{b}\right|\times100.

        """
        pass

    @result
    def result_delta_pal(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ N_TAIL """

    @calculate
    def calculate_n_tail(self, t_min=20, t_max=700, **non_calculation_parameters):
        """
        The number of pTRM tail checks conducted below the maximum temperature used for the best-fit segment on the
        Arai plot (i.e., the number of pTRM tail checks used to analyze the best-fit segment on the Arai plot).
        :param parameter:

        """

        temps = self.tr['temp'].v
        out = [i for i in temps
               if i >= t_min
               if i <= t_max]

        self.results['n_tail'] = len(out)

    @result
    def result_n_tail(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ DRAT TAIL """

    @calculate
    def calculate_drat_tail(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        Maximum absolute difference produced by a pTRM tail check, normalized by the length of the best-fit line
        (Biggin et al., 2007).

        .. math::

           DRAT_{Tail}=\\frac`{\max{\{\left| \delta{tail_i} \right|\}}_{i=1, \ldots, end}}{L}\times{100}

        """

        dtail = self.get_d_tail(t_min=t_min, t_max=t_max)
        max_idx = np.argmax(abs(dtail[component].v))
        dtail = dtail.filter_idx(max_idx)
        L = self.best_fit_line_length(t_min=t_min, t_max=t_max, component=component)
        out = (abs(dtail[component].v) / L) * 100
        self.results['drat_tail'] = out

    @result
    def result_drat_tail(self, rescalc=False, **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ DELTA PAL """

    @calculate
    def calculate_delta_tr(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        Maximum absolute difference produced by a pTRM tail check, normalized by the NRM (obtained from the
        intersection of the best-fit line and the y-axis on an Arai plot; Leonhardt et al., 2004a).

        .. math::

           \delta{TR}=\\frac`{\max{\{\left| \delta{tail_i} \right|\}}_{i=1, \ldots, end}}{\left|Y_{Int.}\right|}\times{100}

        """

        dtail = self.get_d_tail(t_min=t_min, t_max=t_max)
        max_idx = np.argmax(abs(dtail[component].v))
        out = (abs(dtail.filter_idx(max_idx)[component].v) /
               abs(self.result_y_int(t_min=t_min, t_max=t_max, component=component)[0])) * 100
        self.results['delta_tr'] = out

    @result
    def result_delta_tr(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ DELTA PAL """

    @calculate
    def calculate_md_vds(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        Maximum absolute difference produced by a pTRM tail check, normalized by the vector difference sum of the
        NRM (Tauxe and Staudigel, 2004).

        .. math::

           MD_{VDS}=\\frac`{\max{\{\\left| \delta{tail_i} \\right|\}}_{i=1, \ldots, end}}{VDS}\times{100}

        Note
        ----
           Some versions of PmagPy and ThellierGUI use a pTRM tail check statistic called :math:`MD(\%)`. This is identical
           to :math:`MD_{VDS}`, but the change in name emphasizes its calculation method.
        """
        dtail = self.get_d_tail(t_min=t_min, t_max=t_max)
        max_idx = np.argmax(abs(dtail[component].v))
        out = (abs(dtail.filter_idx(max_idx)[component].v) /
               abs(self.calculate_vds())) * 100
        self.results['md_vds'] = out

    @result
    def result_md_vds(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    # """ D_T_STAR """  # todo implement calculate_d_t_star
    #
    # @calculate
    # def calculate_d_t(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
    #     """
    #     The extent of a pTRM tail after correction for angular dependence (Leonhardt et al., 2004a; 2004b).
    #     """
    #     self.results['d_t'] = np.nan
    #
    # @result
    # def result_d_t(self, t_min=20, t_max=700, component=None, recalc=False, **options):
    #     # todo write comment for this method
    #     pass

    """
    Additivity check statistics
    ===========================

    An additivity check is a repeat demagnetization step to test the validity of Thellier's law of additivity
    (Krasa et al., 2003). In the course of a paleointensity experiment, a pTRM at temperature :math:`T_j` is imparted,
    pTRM(:math:`T_j`, :math:`T_0`), where :math:`T_0` is room temperature. An additivity check demagnetizes
    pTRM(:math:`T_j`, :math:`T_0`) by heating to :math:`T_i`, where :math:`T_i < T_j`. The remaining pTRM
    (pTRM(:math:`T_j`, :math:`T_i`)) is subtracted from the previous pTRM acquisition step,
    pTRM(:math:`T_j`, :math:`T_0`), to estimate pTRM:math:`^*`(:math:`T_i`, :math:`T_0`). That is

    .. math::

       pTRM^*(T_i, T_0)=pTRM(T_j, T_0)-pTRM(T_j, T_i)

    where * denotes an estimated value. This estimated value can be compared with a previously observed
    value of pTRM(:math:`T_i`, :math:`T_0`) that was measured earlier in the experiment. The difference
    between the estimated and observed pTRMs is a measure of the violation of
    additivity between :math:`T_i` and :math:`T_0`. The additivity check difference (:math:`AC_{i,j}`)
    is the scalar intensity difference between the two pTRMs:

    .. math::

       AC_{i,j}=pTRM^*(T_i, T_0)-pTRM(T_i, T_0).

    For an additivity check to be included in the analysis, both :math:`T_i` and :math:`T_j`
    must be less than or equal to :math:`T_{max}`.

    """
    ####################################################################################################################
    """ D_T_STAR """

    @calculate
    def calculate_n_ac(self, t_min=20, t_max=700, **non_calculation_parameters):
        """
        The number of additivity checks used to analyze the best-fit segment on the Arai plot
        (i.e., the number of :math:`AC_{i,j}` with :math:`T_i \leq T_{max}` and :math:`T_j \leq T_{max}`).

        :param parameter:

        """
        # check for ac steps
        if not 'ac' in self.data:
            return

        temps = self.ac['temp'].v
        out = [i for i in temps
               if i >= t_min
               if i <= t_max]
        self.results['n_ac'] = len(out)

    @result
    def result_n_ac(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ D_T_STAR """

    @calculate
    def calculate_delta_ac(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        """
        The maximum absolute additivity check difference normalized by the total TRM (obtained from the intersection
        of the best-fit line and the x-axis on an Arai plot; Leonhardt et al., 2004a).

        .. math::

           \\delta{AC}=\\frac{\\max{ \\left\\{ \\left| AC_{i,j} \\right| \\right\\} }_{i \\leq end \\textbf{ and } j \\leq end}}{\\left|X_{Int.}\\right|}\\times{100}.

        """
        # check if ac steps
        if not 'ac' in self.data:
            return
        d_ac = self.get_d_ac(t_min=t_min, t_max=t_max)
        max_idx = np.argmax(abs(d_ac[component].v))
        out = (abs(d_ac.filter_idx(max_idx)[component].v) /
               abs(self.result_x_int(t_min=t_min, t_max=t_max)[0])) * 100
        self.results['delta_ac'] = out

    @result
    def result_delta_ac(self, t_min=20, t_max=700, component='mag', **non_calculation_parameters):
        # todo write comment for this method
        pass

    ####################################################################################################################
    """ HELPER FUNCTIONS FOR CALCULATE METHODS """

    def filter_th_ptrm(self, t_min=20, t_max=700):
        """
        Filters the th and ptrm data so that the temperatures are within t_min, t_max and only temperatures in both
        th and ptrm are returned.
        """
        idx_th_ptrm = np.array([(v1, i1, i2)
                                for i1, v1 in enumerate(self.th['temp'].v)
                                for i2, v2 in enumerate(self.ptrm['temp'].v)
                                if t_min <= v1 <= t_max
                                if t_min <= v2 <= t_max
                                if v1 == v2])

        y = self.th.filter_idx(idx_th_ptrm[:, 1])  # filtered data for t_min t_max
        x = self.ptrm.filter_idx(idx_th_ptrm[:, 2])  # filtered data for t_min t_max
        return y, x

    def calculate_vd(self, t_min=20, t_max=700,
                     **non_calculation_parameter):  # todo move in rockpydata?
        """
        Vector differences

        :param parameter:
        :return:

        """
        idx = (self.th['temp'].v <= t_max) & (t_min <= self.th['temp'].v)
        data = self.th.filter(idx)
        vd = np.array([np.linalg.norm(i) for i in np.diff(data['m'].v, axis=0)])
        return vd

    def calculate_vds(self, **non_calculation_parameter):  # todo move in rockpydata?
        """
        The vector difference sum of the entire NRM vector :math:`\\mathbf{NRM}`.

        .. math::

           VDS=\\left|\\mathbf{NRM}_{n_{max}}\\right|+\\sum\\limits_{i=1}^{n_{max}-1}{\\left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right|}

        where :math:`\\left|\\mathbf{NRM}_{i}\\right|` denotes the length of the NRM vector at the :math:`i^{th}` step.

        Parameters
        ----------
            t_min: float
            t_max: float
            recalc: bool
            non_calculation_parameters: dict
        """
        # tmax = max(self.th['temp'].v)
        # NRM_t_max = self.th.filter(self.th['temp'].v == tmax)['mag'].v[0]
        NRM_t_max = np.linalg.norm(self.th['m'].v[-1])
        NRM_sum = np.sum(np.abs(self.calculate_vd(t_min=0, t_max=700)))
        return abs(NRM_t_max) + NRM_sum

    def x_dash(self, t_min=20, t_max=700, component='mag',
               **non_calculation_parameters):
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

        th, ptrm = self.filter_th_ptrm(t_min=t_min, t_max=t_max)
        x_dash = (th[component].v - self.result_y_int(t_min=t_min, t_max=t_max, component=component)[0])
        x_dash = x_dash / self.result_slope(t_min=t_min, t_max=t_max, component=component)[0]
        x_dash = ptrm[component].v + x_dash
        x_dash = x_dash / 2.

        return x_dash

    def y_dash(self, t_min=20, t_max=700, component='mag',
               **non_calculation_parameters):
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
        th, ptrm = self.filter_th_ptrm(t_min=t_min, t_max=t_max)

        y_dash = th[component].v + \
                 (self.result_slope(t_min=t_min, t_max=t_max, component=component)[0] * ptrm[component].v +
                  self.result_y_int(t_min=t_min, t_max=t_max, component=component)[0])
        y_dash = y_dash / 2
        return y_dash

    def delta_x_dash(self, t_min=20, t_max=700, component='mag',
                     **non_calculation_parameters):
        """

        :math:`\Delta x_0` is the TRM length of the best-fit line on the Arai plot.

        """
        x_dash = self.x_dash(t_min=t_min, t_max=t_max, component=component, **non_calculation_parameters)
        out = abs(np.max(x_dash) - np.min(x_dash))
        return out

    def delta_y_dash(self, t_min=20, t_max=700, component='mag',
                     **non_calculation_parameters):
        """

        :math:`\Delta y_0`  is the NRM length of the best-fit line on the Arai plot.

        """
        y_dash = self.y_dash(t_min=t_min, t_max=t_max, component=component, **non_calculation_parameters)
        out = abs(np.max(y_dash) - np.min(y_dash))
        return out

    def best_fit_line_length(self, t_min=20, t_max=700, component='mag'):
        L = np.sqrt((self.delta_x_dash(t_min=t_min, t_max=t_max, component=component)) ** 2 +
                    (self.delta_y_dash(t_min=t_min, t_max=t_max, component=component)) ** 2)
        return L

    ####################################################################################################################


    """ CHECK SECTION """

    def get_d_ptrm(self, t_min=20, t_max=700, vectorial_diff=False, **non_calculation_parameters):
        """
        The difference between a pTRM check and the original TRM is calculated as the scalar intensity difference

        :math:

           \delta pTRM_{i,j} = pTRM_{check i,j} - TRM_i = pTRM_{check i,j} - TH_i

        :param parameter:
        :return: RockPy data object of CK - PTRM
        """
        data = self._get_ck_data(t_min=t_min, t_max=t_max, vectorial_diff=True)
        new = None
        for temp, d in sorted(data.iteritems()):

            ck_ij, dck_ij, th_i, th_j, ptrm_i, ptrm_j, previous_j = d
            if not new:
                new = deepcopy(dck_ij)
                new['m'] = new['m'].v - ptrm_i['m'].v
                new['mag'] = new['mag'].v - ptrm_i['mag'].v
            else:
                dck_ij['m'] = dck_ij['m'].v - ptrm_i['m'].v
                dck_ij['mag'] = dck_ij['mag'].v - ptrm_i['mag'].v
                new = new.append_rows(dck_ij)

        if vectorial_diff:
            new['mag'] = new.magnitude('m')
        new._row_names = ['dpTRM[%i]' % i for i in new['temp'].v]
        return new

    def get_d_ac(self, t_min=20, t_max=700, **non_calculation_parameters):
        """
        **Additivity check statistics**

        An additivity check is a repeat demagnetization step to test the validity of Thellier's law of additivity
        (Krasa et al., 2003). In the course of a paleointensity experiment, a pTRM at temperature :math:`T_j` is imparted,
        pTRM(:math:`T_j`, :math:`T_0`), where :math:`T_0` is room temperature. An additivity check demagnetizes
        pTRM(:math:`T_j`, :math:`T_0`) by heating to :math:`T_i`, where :math:`T_i < T_j`. The remaining pTRM
        (pTRM(:math:`T_j`, :math:`T_i`)) is subtracted from the previous pTRM acquisition step,
        pTRM(:math:`T_j`, :math:`T_0`), to estimate pTRM:math:`^*`(:math:`T_i`, :math:`T_0`). That is

        .. math::

           pTRM^*(T_i, T_0)=pTRM(T_j, T_0)-pTRM(T_j, T_i)

        where * denotes an estimated value. This estimated value can be compared with a previously observed
        value of pTRM(:math:`T_i`, :math:`T_0`) that was measured earlier in the experiment. The difference
        between the estimated and observed pTRMs is a measure of the violation of
        additivity between :math:`T_i` and :math:`T_0`. The additivity check difference (:math:`AC_{i,j}`)
        is the scalar intensity difference between the two pTRMs:

        .. math::

           AC_{i,j}=pTRM^*(T_i, T_0)-pTRM(T_i, T_0).

        For an additivity check to be included in the analysis, both :math:`T_i` and :math:`T_j`
        must be less than or equal to :math:`T_{max}`.

        """

        ptrm_i0 = self.get_ptrm_i0()

        ac_temps = ptrm_i0['temp'].v
        ptrm_temps = self.ptrm['temp'].v

        # filter indices according to t_min and t_max requirement, also ptrm* and ptrm have to have the same temperature step
        out = np.array([(v, i, i2) for i, v in enumerate(ac_temps) for i2, v2 in enumerate(ptrm_temps)
                        if v == v2
                        if v >= t_min
                        if v <= t_max])

        ptrm_i0_data = ptrm_i0.filter_idx(out[:, 1])
        ptrm_i_data = self.ptrm.filter_idx(out[:, 2])

        d_ac = ptrm_i0_data - ptrm_i_data
        d_ac['time'] = ptrm_i0_data['time'].v
        d_ac['mag'] = d_ac.magnitude(('x', 'y', 'z'))
        return d_ac

    def get_d_tail(self, t_min=20, t_max=700, **non_calculation_parameters):
        """
        pTRM tails check statistics (TR)

        A pTRM tail check is a repeat demagnetization step to test for changes in a specimen's magnetization carried in the
        blocking temperature range above the temperature of the check. The difference between the first NRM measurement and
        the pTRM tail check is calculated as the scalar intensity difference:

        .. math::

           \delta{tail_i} = tail\_check_i - NRM_i = tail\_check_i - y_i,

        where :math:`tail\_check_i` is the pTRM tail check to the :math:`i^{th}` temperature step. The order of the
        difference is such that tail checks smaller than the original NRM yield negative :math:`\delta{tail_i}` and tail
        checks larger than the original NRM give positive :math:`\delta{tail_i}`. For a pTRM tail check to be included
        in the analysis, :math:`T_i` must be less than or equal to :math:`T_{max}`.

        The difference between a pTRM check and the original TRM is calculated as the scalar intensity difference

        :math:

           \delta pTRM_{i,j} = pTRM_{check i,j} - TRM_i = pTRM_{check i,j} - TH_i

        :param parameter:
        :return: RockPy data object of CK - TH
        """

        temps = self.tr['temp'].v
        th_steps = self.th['temp'].v

        out = np.array([(v, i, i2) for i, v in enumerate(temps) for i2, v2 in enumerate(th_steps)
                        if v == v2
                        if v >= t_min
                        if v <= t_max])

        tr_data = self.tr.filter_idx(out[:, 1])
        th_data = self.th.filter_idx(out[:, 2])

        out = tr_data - th_data
        # out['mag'] = out.magnitude(('x', 'y', 'z'))
        return out

    # todo combine ptrm_ij methods
    def get_ck_ptrm_ij(self, ck_data):
        """
        searches for the th_i step before each CK-step
        :return: rpdata
        """

        ck_times = ck_data['time'].v  # times of CK step

        # choose which 0-Field step is used
        th_times = self.th['time'].v  # times of TH steps
        tr_times = self.tr['time'].v  # times of TR steps

        # get index of the th step measured directly before the ac step
        idx = np.array([(i, max([j for j, v2 in enumerate(th_times) if v2 - v < 0])) for i, v in enumerate(ck_times)])
        ck = ck_data.filter_idx(idx[:, 0])
        th_i = self.th.filter_idx(idx[:, 1])

        out = deepcopy(ck)

        for i in ['x', 'y', 'z', 'mag']:
            out[i] = out[i].v - th_i[i].v
        # out['mag'] = out.magnitude('m')

        return out

    def get_ac_ptrm_ij(self):
        """
        Calculates the ptrm_i step from AC-demagnetization steps
        :return: rpdata
        """
        ac_times = self.ac['time'].v  # times of AC step
        th_times = self.th['time'].v  # times of PTRM steps

        # get index of the ptrm step measured directly before the ac step
        idx = np.array([(i, max([j for j, v2 in enumerate(th_times) if v2 - v < 0])) for i, v in enumerate(ac_times)])
        ac_i = self.ac.filter_idx(idx[:, 0])
        th_j = self.th.filter_idx(idx[:, 1])

        ptrm_ij = deepcopy(ac_i)

        for key in ['x', 'y', 'z', 'mag']:
            ptrm_ij[key] = ptrm_ij[key].v - th_j[key].v

        # ptrm_ij['mag'] = ptrm_ij.magnitude(('x', 'y', 'z'))
        return ptrm_ij

    def get_ptrm_i0(self):
        """
        Calculates the ptrm_i step from AC-demagnetization steps
        :return: rpdata
        """
        if not 'ac' in self.data:
            return

        ac_times = self.ac['time'].v  # times of AC step
        ptrm_times = self.ptrm['time'].v  # times of PTRM steps

        # get index of the ptrm step measured directly before the ac step
        idx = np.array([(i, max([j for j, v2 in enumerate(ptrm_times) if v2 - v < 0])) for i, v in enumerate(ac_times)])
        ac_i = self.ac.filter_idx(idx[:, 0])
        ptrm_j = self.ptrm.filter_idx(idx[:, 1])

        ptrm_ij = self.get_ac_ptrm_ij()
        ptrm_i0 = deepcopy(self.get_ac_ptrm_ij())

        for key in ['x', 'y', 'z']:
            ptrm_i0[key] = ptrm_j[key].v - ptrm_i0[key].v
        ptrm_i0['mag'] = ptrm_i0.magnitude(('x', 'y', 'z', 'mag'))
        return ptrm_i0

    def _get_ck_data(self, t_min=20, t_max=700, vectorial_diff=True):
        """
        Helper function, returns the preceding 0-Field steps to each ck step, and calculates the moment gained at ck_ij

        Parameters
        ----------
            vectorial_diff: bool
                if True the vectorial difference is calculated
                if False the scalar difference is calculated
        Returns
        -------
            dict{temp: [ck_ij, dck_ij, th_i, th_j, ptrm_i, ptrm_j, previous_j]}
            where:  ck_ij = the ck step at temperature i after heating to temperature j
                    dck_ij = the vector difference between the ck_ij and previous_j steps
                    th_i   = the th step at temeprtature i
                    th_j   = the th step at temeprtature j
                    ptrm_i = the ptrm step at temeprtature i
                    ptrm_j = the ptrm step at temeprtature j
                    previous_j = the th of tr step at tempertature j depending what protocol is used
         """
        ck_data = {}

        ck = self.ck

        # get maximum th/ptrm temperature, because ck temp has to be lower
        tt_temps = set(self.ptrm['temp'].v) & set(self.th['temp'].v)
        tt_max = max(i for i in tt_temps if i <= t_max)

        # only use steps with t > t_min
        ck = ck.filter(ck['temp'].v >= t_min)
        # only use temps smaller than the last ptrm/th step
        ck = ck.filter(ck['temp'].v < tt_max)

        # The jth step has to come before the ith CK step. Therefore the jth step has a higher temperature
        # get all ck times steps
        ck_ij = zip(ck['time'].v, ck['temp'].v)

        for time, temp in ck_ij:
            # get the correct ckij step
            ck_ij = self.ck.filter(self.ck['temp'].v == temp)
            ck_ij._row_names = ['ck_ij[%i]' % ck_ij['temp'].v[0]]

            ''' j th temperature '''

            # the maximum index of the time before the ith step == the last step done before ckij
            # get only the ptrm in case the last 0-field heating was a th step at a temp higher than ptrm_j
            ptrm_j = max(i for i, v in enumerate(self.ptrm['time'].v) if v < time)

            ptrm_j = self.ptrm.filter_idx(ptrm_j)
            ptrm_j._row_names = ['ptrm_j[%i]' % ptrm_j['temp'].v[0]]  # set rowname for easy reading


            # filter the TH RockPy Data objects for the ptrm_j temperature
            th_j = self.th.filter(self.th['temp'].v == ptrm_j['temp'].v[0])
            th_j._row_names = ['th_j[%i]' % th_j['temp'].v[0]]  # set rowname for easy reading

            # step directly before ck_ij
            prev = (0, None)  # initialize previous (time, dtype)

            for dtype in ['th', 'tr']:
                if not any(i < time for i in self.data[dtype]['time'].v): # check if there are times before time
                    break
                with ignored(KeyError):
                    aux_time = max(v for i, v in enumerate(self.data[dtype]['time'].v) if v < time)
                    if aux_time > prev[0]:  # overwrite prev if there is a 0-Field step that has been measured later
                        prev = (aux_time, dtype)

            self.logger.debug('FOUND << %s >> step before << CK, %i >>' % (prev[1], temp))

            prev_j = self.data[prev[1]].filter(self.data[prev[1]]['time'].v == prev[0])
            prev_j._row_names = ['%s[%i]' % (self.data[prev[1]], self.data[prev[1]]['temp'].v[0])]

            ''' ith temperature '''
            th_i = self.th.filter(self.th['temp'].v == temp)
            th_i._row_names = ['th_i[%i]' % th_i['temp'].v[0]]

            ptrm_i = self.ptrm.filter(self.ptrm['temp'].v == temp)
            ptrm_i._row_names = ['ptrm_i[%i]' % ptrm_i['temp'].v[0]]

            pt_i = self.pt.filter(self.pt['temp'].v == temp)
            pt_i._row_names = ['pt_i[%i]' % pt_i['temp'].v[0]]

            dck_ij = deepcopy(ck_ij)
            dck_ij['m'] = ck_ij['m'].v - prev_j['m'].v
            if vectorial_diff:
                dck_ij['mag'] = dck_ij.magnitude('m')
            else:
                dck_ij['mag'] = np.fabs(dck_ij['mag'].v - prev_j['mag'].v)
            dck_ij._row_names = ['d%s' % i for i in dck_ij._row_names]  # append the d to the row_names

            # put data in list
            data = [ck_ij, dck_ij, th_i, th_j, ptrm_i, ptrm_j, prev_j]
            # add key to ck_data
            ck_data.setdefault(temp, data)
        return ck_data

    def _get_ac_data(self):
        """
        Helper function, returns the preceding th steps to each ck step

        :returns: list [ck_ij, th_i, ptrm_j, th_j]
           where ck_ij = the ptrm check to the ith temperature after heating to the jth temperature
        """
        out = []

        for ac in self.ac:
            th_j = [0, 0, 0, 0, 0]
            for th in self.th:
                if ac[-1] > th[-1]:
                    if th_j[-1] < th[-1]:
                        th_j = th
                if ac[0] == th[0]:
                    th_i = th
            for ptrm in self.ptrm:
                if ptrm[0] == th_j[0]:
                    ptrm_j = ptrm
            for pt in self.pt:
                if pt[0] == th_j[0]:
                    pt_i = pt

            d_ac = pt_i[1:4] - ac[1:4]
            d_ac_m = np.linalg.norm(d_ac)
            d_ac = np.array([ac[0], d_ac[0], d_ac[1], d_ac[2], d_ac_m, ac[-1]])

            out.append([d_ac, th_i, ptrm_j, th_j])
        # for i in out:
        # print i[0][0], i[1][0], i[2][0], i[3][0]
        # print i[0][4], i[1][4], i[2][4], i[3][4]
        return out

    """ EXPORT SECTION """

    def export_tdt(self, folder=None, filename=None):
        import os

        i = ['%s_%.2f_%s' % (t.stype, t.value, t.unit) for t in self.series]
        if not folder:
            folder = os.path.join(os.path.expanduser('~'), 'Desktop')
        if not filename:
            filename = '#'.join([self.sample_obj.name, ';'.join(i), '.tdt'])
        # v = 115.2 #standard volume 8mm diameter
        # v = 4.86E-8  # standard volume 6mm diameter
        volume = 1

        # tdt = {'th': 0.00, 'pt': 0.11, 'ac': 0.14, 'ck': 0.12, 'tr': 0.13} #long version
        tdt = {'th': 0.0, 'pt': 0.1, 'ac': 0.4, 'ck': 0.2, 'tr': 0.3}

        th = deepcopy(self.data['th'])
        pt = deepcopy(self.data['pt'])
        pt = pt.filter_idx(range(1, len(pt['temp'].v)))
        ac = deepcopy(self.data['ac'])
        ck = deepcopy(self.data['ck'])
        tr = deepcopy(self.data['tr'])
        steps = ['th', 'pt', 'ac', 'ck', 'tr']

        out = None

        for idx, i in enumerate([th, pt, ac, ck, tr]):
            i['x'] = i['x'].v / volume
            i['y'] = i['y'].v / volume
            i['z'] = i['z'].v / volume
            i['mag'] = i.magnitude('m')
            i['temp'] = i['temp'].v + tdt[steps[idx]]
            if not out:
                out = i
            else:
                out = out.append_rows(i)
        out = out.sort('time')

        lines = ['Thellier-tdt\r\n', '35.0\t0\t0\t0\t0\r\n']
        for i, v in enumerate(out['m']):
            DIL = general.XYZ2DIL(v[:, 0])
            lines.append(
                '%s\t%.01f\t%.03E\t%.03f\t%.03f\r\n' % (self.sample_obj.name, out['temp'].v[i], DIL[2], DIL[0], DIL[1]))

        with open(os.path.join(folder, filename), 'w+') as f:
            for line in lines:
                f.writelines(line)


if __name__ == '__main__':
    import os.path

    thellier_file = os.path.join(RockPy.test_data_path, 'cryomag', 'NLCRY_Thellier_test.TT')
    s = RockPy.Sample(name='test_sample')
    # m = s.add_measurement(mtype='thellier', machine='cryomag', mfile=thellier_file)
    m = s.add_simulation(mtype='thellier', max_moment=10, noise=1)
    print m.ck
    print m.tr
    # m.plt_thellier()
