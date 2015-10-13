import core.utils

__author__ = 'volk'
import logging
from copy import deepcopy

import RockPy3
import RockPy3.core
import RockPy3.utils
import RockPy3.core.measurement

from RockPy3.core.utils import feature


class Visual(object):
    """
    OPEN QUESTIONS:
       - what if I want to add a non-required feature to the visual e.g. backfield to hysteresis
       - what if I want to have a different visual on a second y axis?
    """
    logger = logging.getLogger('RockPy3.VISUALIZE')
    _required = []

    linestyles = ['-', '--', ':', '-.'] * 100
    marker = ['.', 's', 'o', '+', '*', ',', '1', '3', '2', '4', '8', '<', '>', 'D', 'H', '_', '^',
              'd', 'h', 'p', 'v', '|'] * 100
    colors = ['k', 'b', 'g', 'r', 'm', 'y', 'c'] * 100

    @classmethod
    def implemented_visuals(cls):
        subclasses = set()
        work = [cls]
        while work:
            parent = work.pop()
            for child in parent.__subclasses__():
                if child not in subclasses:
                    subclasses.add(child)
                    work.append(child)
        return {i.__name__.lower(): i for i in subclasses}

    @classmethod
    def get_subclass_name(cls):
        return cls.__name__

    @property
    def calculation_parameter(self):
        if not self._calculation_parameter:
            return dict()
        else:
            return self._calculation_parameter

    @property
    def implemented_features(self):
        out = {i.replace('feature_', ''):
                   getattr(self, i) for i in dir(self) if i.startswith('feature_') if not i.endswith('names')}
        return out

    @property
    def feature_names(self):
        return [i.__name__[8:] for i in self.features]

    def __init__(self, plt_input=None, plt_index=None, fig=None, name=None, coord=None,
                 **options):
        '''
        :param plt_input:
        :param plt_index:
        :param fig:
        :param name:
        :param calculation_parameter:
        :param coord: coordinate system to use ('core', 'geo', 'bed'), use RockPy3.coord otherwise
        :return:
        '''

        calc_params, no_calc_params = core.utils.separate_calculation_parameter_from_kwargs(rpobj=None,
                                                                                            **options)

        self.logger = logging.getLogger('RockPy3.VISUALIZE.' + self.get_subclass_name())
        self.logger.info('CREATING new Visual')

        self._plt_index = plt_index
        self._plt_input = {'sample': [], 'measurement': []}

        if plt_input:
            self.add_plt_input(plt_input=plt_input)

        self._plt_obj = fig

        # set the title: default is the name of the visual
        self.title = no_calc_params.pop('title', self.get_subclass_name())

        self.vdict = {'sample': {},
                      'mobj': {},
                      'visual': {}}

        self.linedict = {}
        self._calculation_parameter = calc_params
        self.init_visual()

    def __getattr__(self, item):
        try:
            return self._calculation_parameter[item]
        except (TypeError, KeyError):
            return object.__getattribute__(self, item)

    def init_visual(self):
        """ this part is needed by every plot, because it is executed automatically """
        self.logger.debug('initializing visual...')

        self.features = []  # list containing all features that have to be plotted for each measurement
        self.single_features = []  # list of features that only have to be plotted one e.g. zero lines

        self.xlabel = 'xlabel'
        self.ylabel = 'ylabel'

    def add_feature(self, features=None):
        self.add_feature_to_list(features=features, feature_list='features')

    def add_feature_to_list(self, feature_list, features=None):
        """
        Adds a feature to the list of feature that will be plotted (self.features)
        """
        list2add = getattr(self, feature_list)

        features = RockPy3.utils.general.to_list(features)  # convert to list if necessary
        # check if feature has been provided, if not show list of implemented features
        if not features:
            self.logger.warning('NO feature selected chose one of the following:')
            self.logger.warning('%s' % sorted(self.implemented_features))
            raise TypeError
        # check if any of the features is not implemented
        if any(feature not in self.implemented_features for feature in features):
            for feature in features:
                if feature not in self.implemented_features:
                    self.logger.warning('FEATURE << %s >> not implemented chose one of the following:' % feature)
                    # remove feature that is not implemented
                    features.remove(feature)
            self.logger.warning('%s' % sorted(self.implemented_features.keys()))

        # check for duplicates and don't add them
        for feature in features:
            if feature not in self.feature_names:
                # add features to self.features
                list2add.append(self.implemented_features[feature])
            else:
                self.logger.info('FEATURE << %s >> already used in %s' % (feature, feature_list))

    def remove_feature(self, features=None):
        self.remove_feature_from_list(feature_list='features', features=features)

    def remove_feature_from_list(self, feature_list, features=None):
        """
        Removes a feature, will result in feature is not plotted

           features:
        :return:
        """
        list2remove = getattr(self, feature_list)
        list_names = [i.__name__[8:] for i in list2remove]

        features = RockPy3.utils.general.to_list(features)  # convert to list if necessary

        # check if feature has been provided, if not show list of implemented features
        if not features:
            self.logger.warning('NO FEATURE SELECTED')
            self.logger.warning('%s' % sorted(self.features))

        # check if any of the features is in used features
        for feature in features:
            if feature in list_names:
                # remove feature that is not implemented
                self.logger.warning('REMOVING feature << %s >>' % feature)
                idx = list_names.index(feature)
                list2remove.remove(list2remove[idx])
                list_names.remove(feature)
            else:
                self.logger.warning('FEATURE << %s >> not used' % feature)

    def add_standard(self):
        """
        Adds standard stuff to plt, like title and x/y labels
        :return:
        """
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    def add_plt_input(self, plt_input):
        plt_input = RockPy3.core.utils.to_list(plt_input)
        for item in plt_input:
            if isinstance(item, RockPy3.Sample):
                self._plt_input['sample'].append(item)
            if isinstance(item, RockPy3.Measurement):
                self._plt_input['measurement'].append(item)

    def get_virtual_study(self):
        """
        creates a virtual study so you can iterate over samplegroups, samples, measurements

        Returns
        -------
           only_measurements: Bool
              True if only measurements are to be plotted
        """
        # initialize
        only_measurements = False
        study = None

        # because iterating over a study, samplegroup is like iterating over a list, I substitute them with lists if not
        # applicable so the plotting is simpler
        if isinstance(self._plt_input, RockPy3.Study):  # input is Study
            study = self._plt_input  # no change
        if isinstance(self._plt_input, RockPy3.SampleGroup):  # input is samplegroup
            study = [self._plt_input]  # list = virtual study
        if isinstance(self._plt_input, RockPy3.Sample):  # input is sample
            study = [[self._plt_input]]  # list(list) = virtual study with a virtual samplegroup
        if type(self._plt_input) in RockPy3.Measurement.inheritors():
            only_measurements = True
            study = [[[self._plt_input]]]
        if isinstance(self._plt_input, list):
            if all(isinstance(item, RockPy3.SampleGroup) for item in self._plt_input):  # all input == samples
                study = self._plt_input
            if all(isinstance(item, RockPy3.Sample) for item in self._plt_input):  # all input == samples
                study = [self._plt_input]

            # all items in _plt_input are measurements
            if all(type(item) in RockPy3.Measurement.inheritors() for item in self._plt_input):
                only_measurements = True
                study = [[self._plt_input]]
        return study, only_measurements

    def plt_visual(self):
        for feature in self.features:
            feature()
        if self.legend.get('show', True):
            self.ax.legend(**self.legend_options)
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    @property
    def legend_options(self):
        """
        takes the self.legend dictionary and only returns the legend compatible parameters
        """
        possible_parameters = ['loc',  # a location code
                               'prop',  # the font property
                               'fontsize	',  # the font size (used only if prop is not specified)
                               'markerscale',  # the relative size of legend markers vs. original
                               'numpoints',  # the number of points in the legend for line
                               'scatterpoints',  # the number of points in the legend for scatter plot
                               'scatteryoffsets',  # a list of yoffsets for scatter symbols in legend
                               'frameon',  # if True, draw a frame around the legend. If None, use rc
                               'fancybox',  # if True, draw a frame with a round fancybox. If None, use rc
                               'shadow',  # if True, draw a shadow behind legend
                               'framealpha',  # If not None, alpha channel for the frame.
                               'ncol',  # number of columns
                               'borderpad',  # the fractional whitespace inside the legend border
                               'labelspacing',  # the vertical space between the legend entries
                               'handlelength',  # the length of the legend handles
                               'handleheight',  # the height of the legend handles
                               'handletextpad',  # the pad between the legend handle and text
                               'borderaxespad',  # the pad between the axes and legend border
                               'columnspacing',  # the spacing between columns
                               'title',  # the legend title
                               'bbox_to_anchor',  # the bbox that the legend will be anchored.
                               'bbox_transform',  # the transform for the bbox. transAxes if None.
                               ]
        options = {i: v for i, v in self.legend.iteritems() if i in possible_parameters}

        # set standards
        if not 'loc' in options:
            options.setdefault('loc', 'best')
        if not 'frameon' in options:
            options.setdefault('frameon', False)
        if not 'numpoints' in options:
            options.setdefault('numpoints', 1)
        if not 'handlelength' in options:
            options.setdefault('handlelength', 0.6)
        if not 'fontsize' in options:
            options.setdefault('fontsize', 12)
        return options

    def normalize_all(self,
                      reference='data', ref_dtype='mag', norm_dtypes='all', vval=None,
                      norm_method='max', norm_factor=None,
                      norm_parameter=False,
                      normalize_variable=False,
                      dont_normalize=None):
        """
        Normalizes all measurements from _required. If _required is empty it will normalize all measurement.
        
        Parameters
        ----------
           reference: str
              reference to normalize to e.g. 'mass', 'th', 'initial_state' ...
           ref_dtype: str
              data type of the reference. e.g. if you want to normalize to the magnetization of the downfield path in a
              hysteresis loop you pur reference = 'downfield', ref_dtype='mag'
           norm_dtypes: list, str
              string or list of strings with the datatypes to be normalized to.
              default: 'all' -> normalizes all dtypes
           vval: flot
              a variable to me normalizes to.
              example: reference='downfield', ref_dtype='mag', vval='1.0', will normalize the
                 data to the downfield branch magnetization at +1.0 T in a hysteresis loop
           norm_method: str
              the method for normalization.
              default: max
           norm_parameter: bool
              if true, instances of the parameter subclass will also be normalized. Only useful in certain situations
              default: False
        """
        study, all_measurements = self.get_virtual_study()
        # cycle through all measurements that will get plotted
        if not self.__class__._required:
            mtypes = None
        else:
            mtypes = self.__class__._required

        for sg_idx, sg in enumerate(study):
            for sample_idx, sample in enumerate(sg):
                if not all_measurements:
                    measurements = sample.get_measurements(mtypes=mtypes)
                else:
                    measurements = study[0][0]
                if not norm_parameter:
                    measurements = [m for m in measurements if
                                    not isinstance(m, RockPy3.Packages.Generic.Measurements.parameters.Parameter)]
                if len(measurements) > 0:
                    for m_idx, m in enumerate(measurements):
                        m.normalize(reference=reference, ref_dtype=ref_dtype,
                                    norm_dtypes=norm_dtypes, norm_method=norm_method,
                                    vval=vval)

    def get_mobj_plt_opt(self, mobj, indices):
        ls, marker, color = self.get_ls_marker_color(indices=indices)

        if mobj.plt_props['linestyle']:
            ls = mobj.plt_props['linestyle']
        if mobj.plt_props['marker']:
            marker = mobj.plt_props['marker']
        if mobj.plt_props['color']:
            color = mobj.plt_props['color']

        return {'linestyle': ls, 'marker': marker, 'color': color}

    def set_mobj_plt_props(self, mobj, indices):
        ls, marker, color = self.get_ls_marker_color(indices=indices)
        if not 'linestyle' in mobj.plt_props:
            mobj.set_plt_prop('linestyle', ls)
        if not 'marker' in mobj.plt_props:
            mobj.set_plt_prop('marker', marker)
        if not 'color' in mobj.plt_props:
            mobj.set_plt_prop('color', color)

    def get_ls_marker_color(self, indices):
        """
        Looks up the appropriate color, marker, ls for given indices
           indices:
        :return:
        """
        if len(indices) == 3:
            return Visual.linestyles[indices[0]], Visual.marker[indices[1]], Visual.colors[indices[2]]
        if len(indices) == 2:
            return Visual.linestyles[indices[0]], Visual.marker[indices[1]], Visual.colors[indices[2]]

    @feature(plt_frequency='single')
    def feature_grid(self, **plt_opt):
        self.ax.grid()

    @feature(plt_frequency='single')
    def feature_zero_lines(self, mobj=None, **plt_opt):
        color = plt_opt.pop('color', 'k')
        zorder = plt_opt.pop('zorder', 0)

        self.ax.axhline(0, color=color, zorder=zorder,
                        **plt_opt)
        self.ax.axvline(0, color=color, zorder=zorder,
                        **plt_opt)

    ####################################################################################################################
    # LEGEND PART

    @property
    def legend(self):
        if not hasattr(self, '_legend'):
            setattr(self, '_legend', dict())
        return getattr(self, '_legend')

    def add_2_legend(self, mobj=None, sample_group=False, sample_name=True,
                     series=True, add_stype=False, add_unit=True,  # related to get_series_lables
                     ):
        """
        adds something to the legend.

        Example
        -------
           feature.add_2_legend(sample_name=True)

        """
        if not mobj:
            if not self.legend:
                self._legend = locals()
            else:
                self._legend.update(locals())
            self.legend.pop('self')

    def get_label_text(self, mobj):
        text = []
        legend = deepcopy(self.legend)
        getter = {'sample_name': mobj.sample_obj.name,
                  'series': mobj.get_series_labels(
                      self.legend.get('series'), self.legend.get('add_stype'), self.legend.get('add_unit'))}

        for type in sorted(legend):
            if legend[type]:
                with RockPy3.ignored(KeyError):
                    text.append(getter[type])
        return ' '.join(text)

    # def feature_result_text(self, mobj, **plt_opt):
    #     RockPy3.Visualize.Features.generic.add_result_text()

    @property
    def ax(self):  # todo change so second y can be implemented
        return RockPy3.Visualize.core.get_subplot(self.fig, self._plt_index)
        # return self._plt_obj.axes[self._plt_index][0]

    @property
    def second_y(self):
        if not self._plt_obj.axes[self._plt_index][1]:
            self._plt_obj.axes[self._plt_index][1] = self._plt_obj.axes[self._plt_index][0].twinx()
        # return self.ax.twinx()
        return self.fig.axes[self._plt_index][1]

    @property
    def second_x(self):
        if not self._plt_obj.axes[self._plt_index][2]:
            self._plt_obj.axes[self._plt_index][2] = self._plt_obj.axes[self._plt_index][0].twinx()
        return self.fig.axes[self._plt_index][2]
        # return self.ax.twinx()

    @property
    def fig(self):
        return self._plt_obj.fig

    ### PLOT PARAMETERS
    def set_plot_parameter(self):
        pass

    def set_xscale(self, value):
        """
        'linear': LinearScale,
        'log':    LogScale,
        'symlog': SymmetricalLogScale

        Parameter
        ---------
           scale:
        """
        # try:
        self.ax.set_xscale(value=value)

    def set_yscale(self, value):
        """
        'linear': LinearScale,
        'log':    LogScale,
        'symlog': SymmetricalLogScale

        Parameter
        ---------
           scale:
        """
        self.ax.set_yscale(value=value)

    def set_ticklabel_format(self, axis='y', style='sci', scilimits=(-2, 2)):
        self.ax.ticklabel_format(axis=axis, style=style, scilimits=scilimits)
        self.ax.xaxis.major.formatter._useMathText = True
        self.ax.yaxis.major.formatter._useMathText = True

    def no_marker(self):
        study, measurements_only = self.get_virtual_study()

        if measurements_only:
            for m in study[0][0]:
                m.marker = ''
        else:
            for sg in study:
                for sample in sg:
                    sample.set_marker = ''
