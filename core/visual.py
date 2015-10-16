import core.utils

__author__ = 'volk'
import logging
import inspect

from copy import deepcopy

import RockPy3
import RockPy3.core
import RockPy3.utils
import RockPy3.core.measurement
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

    possible_text_props = ['x', 'y', 'text', 'rotation',
                           'agg_filter', 'alpha', 'animated', 'axes', 'clip_box', 'clip_on',
                           'clip_path', 'figure', 'fontsize', 'gid', 'label', 'lod',
                           'picker', 'rasterized', 'transform', 'visible', 'zorder',
                           ]
    possible_plt_props = ['agg_filter', 'alpha', 'animated', 'antialiased', 'axes', 'clip_box', 'clip_on', 'clip_path',
                          'color', 'contains', 'dash_capstyle', 'dash_joinstyle', 'dashes', 'drawstyle', 'figure',
                          'fillstyle', 'gid', 'label', 'linestyle', 'linewidth', 'lod', 'marker', 'markeredgecolor',
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

    @property
    def calculation_parameter(self):
        if not self._calculation_parameter:
            return dict()
        else:
            return self._calculation_parameter

    @classmethod
    def implemented_features(cls):
        out = {i.replace('feature_', ''): getattr(cls, i) for i in dir(cls) if i.startswith('feature_') if
               not i.endswith('names')}
        return out

    @property
    def feature_names(self):
        return self.features

    @property
    def plt_props(self):
        return self._plt_props

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

        self.log = logging.getLogger('RockPy3.VISUALIZE.' + self.get_subclass_name())
        self.log.info('CREATING new Visual')

        self._plt_index = plt_index
        self._plt_input = {'sample': [], 'measurement': []}

        # low hierarchy properties are overwritten by measurement props
        self._plt_props = {i: {} for i in self.implemented_features()}
        # higher hierarchy properties, overwrite the measurement props
        self._plt_props_forced = {i: {} for i in self.implemented_features()}

        if plt_input:
            self.add_plt_input(plt_input=plt_input)

        self._RockPy_figure = fig

        # set the title: default is the name of the visual
        self.title = no_calc_params.pop('title', self.get_subclass_name())

        self.vdict = {'sample': {},
                      'mobj': {},
                      'visual': {}}

        self.linedict = {}
        self._calculation_parameter = calc_params
        self.init_visual()
        self.feature_methods = [getattr(self, 'feature_' + name) for name in self.features]
        self.generic_features = []
        self.generic_feature_methods = []

    def __getattr__(self, item):
        try:
            return self._calculation_parameter[item]
        except (TypeError, KeyError):
            return object.__getattribute__(self, item)

    def init_visual(self):
        """ this part is needed by every plot, because it is executed automatically """
        self.log.debug('initializing visual...')

        self.features = []  # list containing all feature names that have to be plotted for each measurement

        self.xlabel = 'xlabel'
        self.ylabel = 'ylabel'

    def add_feature(self, feature=None, **plt_props):
        self.add_feature_to_list(feature=feature)
        self.set_plt_prop(feature=feature, **plt_props)

    def add_feature_to_list(self, feature=None):
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

        # check for duplicates and don't add them
        if feature not in self.features and not 'generic' in feature:
            # generic features can be added multiple times:
            # add features to self.features
            self.features.append(feature)
            self.feature_methods.append(getattr(self, 'feature_' + feature))
        elif 'generic' in feature:
            # create entry in plotproperty dictionary and high hierachy dictionary
            self._plt_props.setdefault(feature + str(len(self.generic_features)), {})
            self._plt_props_forced.setdefault(feature + str(len(self.generic_features)), {})
            self.generic_features.append(feature + str(len(self.generic_features)))
            self.generic_feature_methods.append(getattr(self, 'feature_' + feature))

        else:
            self.log.info('FEATURE << %s >> already used' % (feature))

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
            self.log.warning('%s' % sorted(self.features))

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

    def add_plt_input(self, plt_input):
        plt_input = RockPy3.core.utils.to_list(plt_input)
        for item in plt_input:
            if isinstance(item, RockPy3.RpStudy):
                self._plt_input['sample'].extend([deepcopy(s) for s in item.samplelist])
            if isinstance(item, RockPy3.Sample):
                self._plt_input['sample'].append(deepcopy(item))
            if isinstance(item, RockPy3.Measurement):
                self._plt_input['measurement'].append(deepcopy(item))

    def __call__(self, *args, **kwargs):
        self.add_standard()
        for feature in self.feature_methods:
            feature()
        for idx, generic_feature in enumerate(self.generic_feature_methods):
            generic_feature(idx=idx)

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

    @plot(single=True)
    def feature_grid(self, plt_props=None):
        self.ax.grid(**plt_props)

    @plot(single=True)
    def feature_generic_data(self, plt_props=None):
        # pass
        RockPy3.Packages.Generic.Features.generic.plot_x_y(ax=self.ax, **plt_props)

    @plot(single=True)
    def feature_generic_text(self, plt_props=None):
        RockPy3.Packages.Generic.Features.generic.text_x_y(ax=self.ax, **plt_props)

    @plot(single=True)
    def feature_zero_lines(self, plt_props=None):
        self.ax.axhline(0, **plt_props)
        self.ax.axvline(0, **plt_props)

    ####################################################################################################################
    # LEGEND PART

    @property
    def legend(self):
        if not hasattr(self, '_legend'):
            setattr(self, '_legend', dict())
        return getattr(self, '_legend')

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
    def set_plt_prop(self, feature=None, overwrite_m=False, **prop):
        """
        sets the plot properties for a feature of list of features

        :param feature:
        :param overwrite_m:
        :param prop:
        :return:

        Note
        ----
            'generic' type features can be plotted multiple times -> they get a suffix
        """
        # check if there is a given feature
        if not feature:
            self.log.error('NO FEATURE specified chose one:')
            self.log.error('{}'.format(self.features))

        # iteerate through properties passed
        for p, value in prop.items():
            if p in self.possible_plt_props or p in self.possible_text_props:
                feature = RockPy3.core.utils.to_list(feature)
                for f in feature:
                    if f in self.implemented_features():
                        if 'generic' in f:
                            f += str(len(self.generic_features) - 1)
                        try:
                            self.log.debug(
                                'CHANGING plot property {}.{} from {} -> {}'.format(f, p, self.plt_props[f][p], value))
                        except KeyError:
                            self.log.debug('Setting plot property {}.{} to {}'.format(f, p, value))
                        if overwrite_m:
                            self._plt_props_forced[f].setdefault(p, value)
                        else:
                            self.plt_props[f].setdefault(p, value)
            else:
                self.log.warning('PROPERTY << {} >> not in Line2D properties nor text properties'.format(p))

    def set_xscale(self, value):
        """
        'linear': LinearScale,
        'log':    LogScale,
        'symlog': SymmetricalLogScale

        Parameter
        ---------
           scale:
        """
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


def set_colorscheme(scheme):
    RockPy3.colorscheme = RockPy3.core.utils.colorscheme(scheme)
    return RockPy3.core.utils.colorscheme(scheme)
