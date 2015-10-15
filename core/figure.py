__author__ = 'mike'

import logging
import os
import inspect

import numpy as np

from matplotlib import pyplot as plt

import RockPy3
import RockPy3.core.utils


# from RockPy3.core.visual import Visual


class Figure(object):
    def __init__(self, title=None, figsize=(5, 5)):  # todo size of figure
        """
        Container for visuals.

        Parameters
        ----------
           title: str
              title text written on top of figure

        """
        self.log = logging.getLogger(__name__)
        self.log.info('CREATING new figure')
        self.logged_implemented = False
        self.__log_implemented(ignore_once=False)
        # create dictionary for visuals {visual_name:visual_object}
        self._visuals = []
        self._n_visuals = 0
        self.xsize = figsize[0]
        self.ysize = figsize[1]
        self.fig = plt.figure()  # initialize figure
        self.title = title

    @property
    def visuals(self):
        return self._visuals

    def __getitem__(self, item):
        """
        Asking for an item from the plot will result in one of two things.
           1. If you ask for a name or an index, and the name or index are
        :param item:
        :return:
           Visual Object
        """
        # try:
        if item in self.vtypes:
            idx = [i for i, v in enumerate(self.vtypes) if v == item]
            return [self._visuals[i][2] for i in idx]
        if item in self.vnames:
            idx = self.vnames.index(item)
            return self._visuals[idx][2]
        if type(item) == int:
            return self._visuals[item][2]
        else:
            raise KeyError('%s can not be found' % item)

    def add_visual(self, visual, name=None, plt_input=None,
                   **visual_opt):
        """
        adds a visual to the plot. This creates a new subplot.

        Parameters
        ----------

           visual: list, str
              name of visual to add.

        """
        # calculation_parameters, kwargs = RockPy3.core.utils.separate_calculation_parameter_from_kwargs(rpobj=self, **visual_opt)
        # convert visual to list
        visuals = RockPy3.core.utils.to_list(visual)
        # for easy checking convert the names to lower case
        visuals = map(str.lower, visuals)

        for visual in visuals:
            # check if visual exists otherwise don't create it
            if visual in RockPy3.implemented_visuals:
                self.log.debug('VISUAL << %s > found' % visual)
                self.__log_implemented()
                if not name:
                    name = visual
                n = self._n_visuals
                # create instance of visual by dynamically calling from implemented_visuals dictionary
                visual_obj = RockPy3.implemented_visuals[visual](plt_input=plt_input, plt_index=n, fig=self, name=name,
                                                                 **visual_opt)
                self._visuals.append([name, visual, visual_obj])
                self._n_visuals += 1
            else:
                self.log.warning('VISUAL << %s >> not implemented yet' % visual)
                self.__log_implemented(ignore_once=True)
                return
        return visual_obj

    def __log_implemented(self, ignore_once=False):
        if not self.logged_implemented or ignore_once:
            self.log.info('-'.join('' for i in range(70)))
            self.log.info('IMPLEMENTED \tVISUALS: features')
            for n, v in RockPy3.implemented_visuals.items():
                self.log.info('\t{}:\t{}'.format(n, ', '.join(v.implemented_features().keys())))
            self.log.info('-'.join('' for i in range(70)))
            if not ignore_once:
                self.logged_implemented = True

    @property
    def vnames(self):
        return [i[0] for i in self._visuals]

    @property
    def vtypes(self):
        return [i[1] for i in self._visuals]

    @property
    def vinstances(self):
        return [i[2] for i in self._visuals]

    def plt_all(self):
        """
        helper function calls plt_visual for each visual
        :return:
        """
        self.fig, self.axes = self._create_fig(xsize=self.xsize, ysize=self.ysize)
        for name, type, visual in self._visuals:
            visual.plt_visual()

    def _create_fig(self, xsize=5, ysize=5):
        """
        Wrapper that creates a new figure but first deletes the old one.
        """
        # create new figure with appropriate number of subplots
        return generate_plots(n=self._n_visuals, xsize=xsize, ysize=ysize)

    def get_xylims(self, visuals=None):
        xlim = []
        ylim = []
        # cycle throu visuals to get
        if not visuals:
            visuals = self.vinstances

        for visual in visuals:
            xlim.append(visual.ax.get_xlim())
            ylim.append(visual.ax.get_ylim())

        xlim = [min([i[0] for i in xlim]), max([i[1] for i in xlim])]
        ylim = [min([i[0] for i in ylim]), max([i[1] for i in ylim])]
        return xlim, ylim

    def show(self,
             set_xlim=None, set_ylim=None,
             equal_lims=False,
             save_path=None,
             pad=0.4, w_pad=0.5, h_pad=1.0,
             **options):
        """
        calls all visuals

        Raises
        ------
            TypeError if no visuals have been added
        """
        if not self.visuals:
            self.log.error('NO VISUALS ADDED! Please add any of the followig visuals:')
            for visual in sorted(Visual.implemented_visuals()):
                self.log.info('\t %s' % visual)
            raise TypeError('add a visual')

        # actual plotting of the visuals
        self.plt_all()

        if set_xlim == 'equal' or set_ylim == 'equal' or equal_lims:
            xlim, ylim = self.get_xylims()

            # cycle through visuals to set
            for name, type, visual in self._visuals:
                if set_xlim == 'equal' or equal_lims:
                    visual.ax.set_xlim(xlim)
                if set_ylim == 'equal' or equal_lims:
                    visual.ax.set_ylim(ylim)

        # check if two entries and each is float or int
        if set_xlim:
            if len(set_xlim) == 2 and any(isinstance(i, (float, int)) for i in set_xlim):
                for name, type, visual in self._visuals:
                    visual.ax.set_xlim(set_xlim)
        # check if two entries and each is float or int
        if set_ylim:
            if len(set_ylim) == 2 and any(isinstance(i, (float, int)) for i in set_ylim):
                for name, type, visual in self._visuals:
                    visual.ax.set_ylim(set_ylim)

        # plt.tight_layout()

        if self.title:
            self.fig.suptitle(self.title, fontsize=20)
            # (left, bottom, right, top) in the normalized figure coordinate that the whole subplots area
            # (including labels) will fit into
            self.fig.set_tight_layout(tight={'rect': (0, 0, 1, 0.95)})

        self.fig.set_tight_layout(tight={'pad': pad, 'w_pad': w_pad, 'h_pad': h_pad})

        if save_path:
            if save_path == 'Desktop':
                file_name = os.path.basename(inspect.stack()[-1][1])
                file_name += options.get('append', '')
                save_path = os.path.join(os.path.expanduser('~'), 'Desktop', file_name + '.pdf')
            plt.savefig(save_path)
        else:
            plt.show()


def generate_plots(n=3, xsize=5., ysize=5., tight_layout=False):
    """
    Generates a number of subplots that are organized in a way to fit on a landscape plot.

    Parameter
    ---------
       n: int
          number of plots to be generated
       xsize: float
          size along x for each plot
       ysize: float
          size along y for each plot
       tight_layout: bool
          using tight_layout (True) or not (False)

    Returns
    -------
       fig matplotlib figure instance
    """
    a = np.floor(n ** 0.5).astype(int)
    b = np.ceil(1. * n / a).astype(int)
    # print "a\t=\t%d\nb\t=\t%d\na*b\t=\t%d\nn\t=\t%d" % (a,b,a*b,n)
    fig = plt.figure(figsize=(xsize * b, ysize * a), tight_layout=True)

    axes = []

    for i in range(1, n + 1):
        ax1 = fig.add_subplot(a, b, i)
        for ax in [ax1]:
            ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
            ax.xaxis.major.formatter._useMathText = True
            ax.yaxis.major.formatter._useMathText = True
        axes.append([ax1, None, None])
    fig.set_tight_layout(tight_layout)
    return fig, axes
