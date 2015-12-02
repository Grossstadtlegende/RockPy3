__author__ = 'volk'
import logging
import inspect
from copy import deepcopy
import RockPy3
import RockPy3.core
import RockPy3.utils
import RockPy3.core.measurement
import RockPy3.Packages.Generic.Features.generic
from RockPy3.core.utils import plot, to_list


class Visual(object):
    """
    OPEN QUESTIONS:
       - what if I want to add a non-required feature to the visual e.g. backfield to hysteresis
       - what if I want to have a different visual on a second y axis?
    """
    log = logging.getLogger('RockPy3.VISUALIZE')
    required = []

    linestyles = ['-', '--', ':', '-.'] * 100
    marker = ['.', 's', 'o', '+', '*', ',', '1', '3', '2', '4', '8', '<', '>', 'D', 'H', '_', '^',
              'd', 'h', 'p', 'v', '|'] * 100
    colors = ['k', 'b', 'g', 'r', 'm', 'y', 'c'] * 100

    possible_text_props = ['x', 'y', 's', 'rotation', 'size',
                           'agg_filter', 'alpha', 'animated', 'axes', 'clip_box', 'clip_on',
                           'clip_path', 'figure', 'fontsize', 'gid', 'label', 'lod',
                           'picker', 'rasterized', 'transform', 'visible', 'zorder',
                           ]
    possible_plt_props = ['agg_filter', 'alpha', 'animated', 'antialiased', 'axes', 'clip_box', 'clip_on', 'clip_path',
                          'color', 'contains', 'dash_capstyle', 'dash_joinstyle', 'dashes', 'drawstyle', 'figure',
                          'fillstyle', 'gid', 'label', 'linestyle', 'linewidth', 'lw', 'lod', 'marker',
                          'markeredgecolor',
                          'markeredgewidth', 'markerfacecolor', 'markerfacecoloralt', 'markersize', 'markevery',
                          'path_effects', 'picker', 'pickradius', 'rasterized', 'sketch_params', 'snap',
                          'solid_capstyle', 'solid_joinstyle', 'transform', 'url', 'visible', 'xdata', 'ydata',
                          'zorder',
                          ]

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

    @classmethod
    def implemented_features(cls):
        out = {i.replace('feature_', ''): getattr(cls, i) for i in dir(cls) if i.startswith('feature_') if
               not i.endswith('names')}
        return out

    @property
    def feature_names(self):
        return self.standard_features

    @property
    def plt_props(self):
        return self._plt_props

    @classmethod
    def separate_plt_props_from_kwargs(cls, **kwargs):

        plt_props = {}
        txt_props = {}
        for k in list(kwargs.keys()):
            if k in cls.possible_plt_props:
                plt_props.setdefault(k, kwargs.pop(k))
            if k in cls.possible_text_props:
                txt_props.setdefault(k, kwargs.pop(k))
        return plt_props, txt_props, kwargs

    def __init__(self, visual_input=None, plt_index=None, fig=None, name=None, coord=None,
                 plot_groupmean=None, plot_samplemean=None, plot_samplebase=None, plot_groupbase=None,
                 plot_other=None,
                 base_alpha=None, ignore_samples=None, result_from_means=None,
                 xlabel=None, ylabel=None, xlims=None, ylims=None,
                 xscale=None, yscale=None,
                 **options):
        '''
        :param visual_input:
        :param plt_index:
        :param fig:
        :param name:
        :param calculation_parameter:
        :param coord: coordinate system to use ('core', 'geo', 'bed'), use RockPy3.coord otherwise
        :return:
        '''

        plt_props, txt_props, options = self.separate_plt_props_from_kwargs(**options)
        calc_params, no_calc_params = RockPy3.core.utils.separate_calculation_parameter_from_kwargs(rpobj=None,
                                                                                                    **options)
        self.log = logging.getLogger('RockPy3.VISUALIZE.' + self.get_subclass_name())
        self.log.info('CREATING new Visual')

        self._plt_index = plt_index
        self._visual_input = []

        # low hierarchy properties are overwritten by measurement props
        self._plt_props = plt_props
        self._txt_props = txt_props
        self.features = {}

        self._visual_input = RockPy3.core.utils.sort_plt_input(visual_input)
        self._RockPy_figure = fig

        # set the title: default is the name of the visual
        self.title = no_calc_params.pop('title', self.get_subclass_name())

        self.linedict = {}
        self.calculation_parameter = calc_params

        ### calculation parameter DEBUG
        self.log.debug('Using calculation_parameters:')

        for k, v in calc_params.items():
            self.log.debug('{:30}: {}'.format(k, v))

        self.init_visual()

        if xlabel:
            self.xlabel = xlabel
        if ylabel:
            self.ylabel = ylabel

        for feature in self.standard_features:
            self.add_feature(feature=feature)

        # inputsection
        self.plot_groupmean, self.plot_samplemean, self.plot_groupbase, self.plot_samplebase, self.plot_other = plot_groupmean, plot_samplemean, plot_groupbase, plot_samplebase, plot_other
        self.result_from_means = result_from_means
        self.base_alpha = base_alpha
        self.ignore_samples = ignore_samples

        self.xlims = xlims
        self.ylims = ylims

        # scales
        self.xscale = xscale
        self.yscale = yscale

    def __getattr__(self, item):
        try:
            return self.calculation_parameter[item]
        except (TypeError, KeyError):
            return object.__getattribute__(self, item)

    def init_visual(self):
        """ this part is needed by every plot, because it is executed automatically """
        self.log.debug('initializing visual...')

        self.standard_features = []  # list containing all feature names that have to be plotted for each measurement

        self.xlabel = 'xlabel'
        self.ylabel = 'ylabel'

    def add_feature(self, feature=None, feature_input=None,
                    plot_groupmean=None, plot_samplemean=None, plot_samplebase=None, plot_groupbase=None,
                    plot_other=None, base_alpha=None,
                    ignore_samples=False, result_from_means=None,
                    **plt_props):
        """
        Adds a feature to the visual. Each feature may have multiple instances, and can but does not have to have a
        separate input from the visual input.
        """
        calculation_parameter, plt_props = RockPy3.core.utils.separate_calculation_parameter_from_kwargs(**plt_props)

        new_feature_name = self.add_feature_to_dict(feature=feature)
        self.set_plt_prop(feature_name=new_feature_name, **plt_props)
        self.features[new_feature_name]['feature_input'] = RockPy3.core.utils.sort_plt_input(feature_input)
        for k, v in (('base_alpha', base_alpha), ('ignore_samples', ignore_samples),
                     ('calculation_parameter', calculation_parameter),
                     ('plot_groupmean', plot_groupmean), ('plot_samplemean', plot_samplemean),
                     ('plot_groupbase', plot_groupbase), ('plot_samplebase', plot_samplebase),
                     ('plot_other', plot_other),
                     ('result_from_means', result_from_means),
                     ):
            self.features[new_feature_name].setdefault(k, v)

    def add_feature_to_dict(self, feature=None):
        """
        Adds a feature to the list of feature that will be plotted (self.features)
        """
        # check if feature has been provided, if not show list of implemented features
        if not feature:
            self.log.warning('NO feature selected chose one of the following:')
            self.log.warning('%s' % sorted(self.implemented_features().keys()))
            raise TypeError

        # check if any of the features is not implemented
        if feature not in self.implemented_features():
            self.log.warning('FEATURE << %s >> not implemented chose one of the following:' % feature)
            self.log.warning('%s' % sorted(self.implemented_features().keys()))

        new_feature_name = '{}_{:03}'.format(feature, len(self.features))
        # create entry in plotproperty dictionary and high hierachy dictionary
        self.features.setdefault(new_feature_name, {})
        self.features[new_feature_name].setdefault('feature_props', {})
        self.features[new_feature_name].setdefault('feature_input', [])
        self.features[new_feature_name].setdefault('method', None)
        self.features[new_feature_name]['method'] = getattr(self, 'feature_' + feature)
        return new_feature_name

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
            self.log.warning('NO FEATURE SELECTED')
            self.log.warning('%s' % sorted(self.standard_features))

        # check if any of the features is in used features
        for feature in features:
            if feature in list_names:
                # remove feature that is not implemented
                self.log.warning('REMOVING feature << %s >>' % feature)
                idx = list_names.index(feature)
                list2remove.remove(list2remove[idx])
                list_names.remove(feature)
            else:
                self.log.warning('FEATURE << %s >> not used' % feature)

    def add_standard(self):
        """
        Adds standard stuff to plt, like title and x/y labels
        :return:
        """
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    def add_input(self, visual_input, groupmean=True, samplemean=True, base=True, other=True, base_alpha=0.5,
                  ignore_samples=False):
        """
        Parameters
        ----------
            visual_input: study, sample or list of studies, measurement or list of measurements
                the measurements you want to have plotted
            ignore_mean: bool
                default: False
                if False: the study / sample is searched for mean_measurements.
                    The base measurements are not added to the visual_input
        """

        self._visual_input = RockPy3.core.utils.add_to_plt_input(plt_input=visual_input, to_add_to=self.visual_input,
                                                                 groupmean=groupmean, samplemean=samplemean, base=base,
                                                                 other=other)

    def normalize(self, reference='data', ref_dtype='mag', norm_dtypes='all', vval=None,
                  norm_method='max', norm_factor=None, result=None,
                  normalize_variable=False, dont_normalize=None,
                  norm_initial_state=True, **options):

        # deepcopy the input, in case its a copy from fig_input
        self._visual_input = deepcopy(self.visual_input)

        for plt_type, mlist in self.visual_input.items():
            for m in mlist:
                if isinstance(m, RockPy3.Parameter):
                    continue
                m.normalize(reference=reference, ref_dtype=ref_dtype, norm_dtypes=norm_dtypes, vval=vval,
                            norm_method=norm_method, norm_factor=norm_factor, result=result,
                            normalize_variable=normalize_variable, dont_normalize=dont_normalize,
                            norm_initial_state=norm_initial_state, **options)
        for f in self.features:
            for plt_type, mlist in self.features[f]['feature_input'].items():
                for m in mlist:
                    for m in self.features[f]['feature_input']:
                        m.normalize(reference=reference, ref_dtype=ref_dtype, norm_dtypes=norm_dtypes, vval=vval,
                                    norm_method=norm_method, norm_factor=norm_factor, result=result,
                                    normalize_variable=normalize_variable, dont_normalize=dont_normalize,
                                    norm_initial_state=norm_initial_state, **options)

    def __call__(self, *args, **kwargs):
        self.add_standard()
        for feature in self.features:
            self.features[feature]['method'](name=feature)
        if self.show_legend():
            self.ax.legend(**self.legend_options)

    def show_legend(self):
        # if self.legend_options['show']:
        #     return True
        mlist = [m for k, v in self.visual_input.items() for m in v]
        if any(m.plt_props['label'] != '' for m in mlist):
            return True
        for f in self.features:
            mlist = [m for k, v in self.features[f]['feature_input'].items() for m in v]
            if any(m.plt_props['label'] != '' for m in mlist):
                return True
        else:
            return False

    @property
    def visual_input(self):
        if not any(self._visual_input[plt_type] for plt_type in self._visual_input):
            self._visual_input = self._RockPy_figure.fig_input
        return self._visual_input

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
        options = {i: v for i, v in self.legend.items() if i in possible_parameters}

        # set standards
        if not 'loc' in options:
            options.setdefault('loc', 'best')
        if not 'frameon' in options:
            options.setdefault('frameon', False)
        if not 'numpoints' in options:
            options.setdefault('numpoints', 1)
        if not 'handlelength' in options:
            options.setdefault('handlelength', 1)
        if not 'fontsize' in options:
            options.setdefault('fontsize', 12)
        return options

    @plot(single=True)
    def feature_grid(self, plt_props=None):
        plt_props['color'] = 'k'
        plt_props['linestyle'] = '--'
        self.ax.grid(**plt_props)

    @plot(single=True)
    def feature_generic_data(self, plt_props=None):
        RockPy3.Packages.Generic.Features.generic.plot_x_y(ax=self.ax, **plt_props)

    @plot(single=True)
    def feature_generic_text(self, plt_props=None):
        RockPy3.Packages.Generic.Features.generic.text_x_y(ax=self.ax, **plt_props)

    @plot(single=True)
    def feature_zero_lines(self, plt_props=None):
        plt_props['color'] = 'k'
        plt_props['marker'] = ''
        plt_props['linestyle'] = '-'
        self.ax.axhline(0, **plt_props)
        self.ax.axvline(0, **plt_props)

    ####################################################################################################################
    # LEGEND PART

    @property
    def legend(self):
        if not hasattr(self, '_legend'):
            setattr(self, '_legend', dict())
        return getattr(self, '_legend')

    def remove_labels(self, remove_from_features=True):
        'removes all labels from all measurements'
        for in_type in self.visual_input:
            for m in self.visual_input[in_type]:
                m.set_plt_prop('label', '')
        if remove_from_features:
            for f in self.features:
                for in_type in ('groupmean', 'samplemean', 'groupbase', 'samplebase', 'other'):
                    for m in self.features[f]['feature_input'][in_type]:
                        m.set_plt_prop('label', '')

    @property
    def ax(self):
        return self._RockPy_figure.axes[self._plt_index][0]

    @property
    def second_y(self):
        if not self._RockPy_figure.axes[self._plt_index][1]:
            self._RockPy_figure.axes[self._plt_index][1] = self._RockPy_figure.axes[self._plt_index][0].twiny()
        return self.fig.axes[self._plt_index][1]

    @property
    def second_x(self):
        if not self._RockPy_figure.axes[self._plt_index][2]:
            self._RockPy_figure.axes[self._plt_index][2] = self._RockPy_figure.axes[self._plt_index][0].twinx()
        return self._RockPy_figure.axes[self._plt_index][2]

    @property
    def fig(self):
        return self._RockPy_figure.fig

    ### PLOT PARAMETERS
    def set_plt_prop(self, feature=None, feature_name=None, **prop):
        """
        sets the plot properties for a feature of list of features

        Parameters
        ----------
            feature: str
                providing a feature, it will set the feature_plt_props for all features of that type.
                e.g. feature: hysteresis will set the props for all hysteresis features, if there are more than one.
            feature_name: str
                setting the properties for a specific feature e.g. hysteresis_001 but not hysteresis_002
            prop: kwargs containing plot properties

        Note
        ----
            'generic' type features can be plotted multiple times -> they get a suffix
        """
        # check if there is a given feature
        if not feature and not feature_name:
            self.log.error('NO FEATURE specified chose one:')
            self.log.error('{}'.format(list(self.features.keys())))

        # iteerate through properties passed
        for p, value in prop.items():
            if p in self.possible_plt_props or p in self.possible_text_props:
                if feature and feature in self.implemented_features():
                    features = [f for f in self.feautes if feature in f]
                    for f in features:
                        try:
                            self.log.debug(
                                'CHANGING plot property {}.{} from {} -> {}'.format(feature, p,
                                                                                    self.plt_props[feature][p], value))
                        except KeyError:
                            self.log.debug('Setting plot property {}.{} to {}'.format(feature, p, value))
                        else:
                            self.features[f]['feature_props'].setdefault(p, value)
                if feature_name:
                    self.features[feature_name]['feature_props'].setdefault(p, value)

            else:
                self.log.warning('PROPERTY << {} >> not in Line2D properties nor text properties'.format(p))

    def set_ticklabel_format(self, axis='y', style='sci', scilimits=(-2, 2)):
        self.ax.ticklabel_format(axis=axis, style=style, scilimits=scilimits)
        self.ax.xaxis.major.formatter._useMathText = True
        self.ax.yaxis.major.formatter._useMathText = True

# class GenericData(Visual):
#
#     def __init__(self, xdata, ydata, plt_index=None, fig=None, name=None,
#                  xlabel=None, ylabel=None,
#                  **options):
#
#         plt_props, txt_props, options = self.separate_plt_props_from_kwargs(**options)
#
#         self.log = logging.getLogger('RockPy3.VISUALIZE.' + self.get_subclass_name())
#         self.log.info('CREATING new Visual')
#
#         self._plt_index = plt_index
#         self._visual_input = []
#
#         # low hierarchy properties are overwritten by measurement props
#         self._plt_props = plt_props
#         self._txt_props = txt_props
#         self.features = {}
#
#         self._RockPy_figure = fig
#
#         # set the title: default is the name of the visual
#         self.title = options.pop('title', self.get_subclass_name())
#
#         self.linedict = {}
#
#         ### calculation parameter DEBUG
#         self.log.debug('Using calculation_parameters:')
#         self.init_visual()
#
#         if xlabel:
#             self.xlabel = xlabel
#         if ylabel:
#             self.ylabel = ylabel
#
#         for feature in self.standard_features:
#             self.add_feature(feature=feature)
#
#     def __call__(self)
#         self.add_standard()
#         for feature in self.features:
#             self.features[feature]['method'](name=feature)
#
#         if self.show_legend():
#             self.ax.legend(**self.legend_options)
#
#     def init_visual(self):
#         self.standard_features = ['generic_data']
#         self.xlabel = ''
#         self.ylabel = ''

def set_colorscheme(scheme):
    RockPy3.colorscheme = RockPy3.core.utils.colorscheme(scheme)
    return RockPy3.core.utils.colorscheme(scheme)
