__author__ = 'mike'

import logging
import os
import inspect
import numpy as np
import RockPy3
import RockPy3.core.utils
import matplotlib.pyplot as plt
from RockPy3.core.visual import Visual


class Figure(object):
    def __init__(self, title=None, figsize=(5, 5), columns=None, tightlayout=True,
                 fig_input=None,
                 plot_groupmean=None, plot_samplemean=None, plot_samplebase=None, plot_groupbase=None,
                 plot_other=None,
                 base_alpha=None, ignore_samples=None, result_from_means=None,
                 **fig_props
                 ):
        """
        Container for visuals.

        Parameters
        ----------
           title: str
              title text written on top of figure

        """
        self.plt_props, self.txt_props, kwargs = Visual.separate_plt_props_from_kwargs(**fig_props)

        self.log = logging.getLogger(__name__)
        self.log.info('CREATING new figure')
        self.logged_implemented = False
        self.__log_implemented(ignore_once=False)
        # create dictionary for visuals {visual_name:visual_object}
        self._visuals = []
        self._n_visuals = 0
        self.columns = columns
        self.tightlayout = tightlayout
        self.xsize, self.ysize = figsize
        self._fig = None
        self.title = title

        self.fig_input = RockPy3.core.utils.sort_plt_input(fig_input)

        mlist, mean_list = RockPy3.core.utils.mlist_from_plt_input(fig_input)
        self.calculation_parameter, kwargs = RockPy3.core.utils.separate_calculation_parameter_from_kwargs(
            mlist=mlist.extend(mean_list), **kwargs)

        self.plot_groupmean, self.plot_samplemean, self.plot_groupbase, self.plot_samplebase, self.plot_other = plot_groupmean, plot_samplemean, plot_groupbase, plot_samplebase, plot_other
        self.result_from_means = result_from_means
        self.base_alpha = base_alpha
        self.ignore_samples = ignore_samples

    def add_input(self, fig_input, groupmean=True, samplemean=True, base=True, other=True):
        self.fig_input = RockPy3.core.utils.add_to_plt_input(plt_input=fig_input, to_add_to=self.fig_input,
                                                             groupmean=groupmean, samplemean=samplemean, base=base,
                                                             other=other)

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

    def add_visual(self, visual, name=None, visual_input=None,
                   plot_groupmean=None, plot_samplemean=None, plot_samplebase=None, plot_groupbase=None,
                   plot_other=None, base_alpha=None, result_from_means=None,
                   xlabel=None, ylabel=None,
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
                visual_obj = RockPy3.implemented_visuals[visual](
                    visual_input=visual_input, plt_index=n, fig=self, name=name,
                    plot_groupmean=plot_groupmean, plot_groupbase=plot_groupbase,
                    plot_samplemean=plot_samplemean, plot_samplebase=plot_samplebase,
                    plot_other=plot_other, base_alpha=base_alpha, result_from_means=result_from_means,
                    xlabel=xlabel, ylabel=ylabel,
                    **visual_opt)
                self._visuals.append([name, visual, visual_obj])
                self._n_visuals += 1
            else:
                self.log.warning('VISUAL << %s >> not implemented yet' % visual)
                self.__log_implemented(ignore_once=True)
                return
        return visual_obj

    def __log_implemented(self, ignore_once=False):
        """
        short logging function to show what is implemented
        :param ignore_once:
        :return:
        """
        if not self.logged_implemented or ignore_once:
            self.log.info('-'.join('' for i in range(70)))
            self.log.info('IMPLEMENTED \tVISUALS: features')
            for n, v in sorted(RockPy3.implemented_visuals.items()):
                self.log.info('\t{}:\t{}'.format(n.upper(), ', '.join(v.implemented_features().keys())))
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
        for name, type, visual in self._visuals:
            visual()

    def _create_fig(self, xsize=5, ysize=5):
        """
        Wrapper that creates a new figure but first deletes the old one.
        """
        # create new figure with appropriate number of subplots
        if self._fig:
            plt.close(self._fig)
        return generate_plots(n=self._n_visuals, xsize=xsize, ysize=ysize, columns=self.columns,
                              tight_layout=self.tightlayout)

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
             equal_lims=False, center_lims=False,
             save_path=None,
             pad=0.4, w_pad=0.5, h_pad=1.0,
             file_name=None,
             **options):
        """
        calls all visuals

        Raises
        ------
            TypeError if no visuals have been added
        """

        self._fig, self.axes = self._create_fig(xsize=self.xsize, ysize=self.ysize)

        if not self.visuals:
            self.log.error('NO VISUALS ADDED! Please add any of the followig visuals:')
            for visual in sorted(Visual.implemented_visuals()):
                self.log.info('\t %s' % visual)
            raise TypeError('add a visual')

        # actual plotting of the visuals
        self.plt_all()


        for name, type, visual in self._visuals:
            if visual.xlims:
                visual.ax.set_xlim(visual.xlims)
            if visual.ylims:
                visual.ax.set_ylim(visual.ylims)
            else:
                xlim = visual.ax.get_xlim()
                ylim = visual.ax.get_ylim()
                visual.ax.set_xlim([xlim[0]-xlim[1]*0.05, xlim[1]+xlim[1]*0.05])

        if set_xlim == 'equal' or set_ylim == 'equal' or equal_lims:
            if center_lims:
                xl = max(np.abs(xlim))
                yl = max(np.abs(ylim))
                xlim = [-xl, xl]
                ylim = [-yl, yl]

            # cycle through visuals to set
            for name, type, visual in self._visuals:
                if set_xlim == 'equal' or equal_lims:
                    visual.ax.set_xlim(xlim)
                if set_ylim == 'equal' or equal_lims:
                    visual.ax.set_ylim(ylim)

        for name, type, visual in self._visuals:
            visual.ax.legend(**visual.legend_options)

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

        self._fig.set_tight_layout(tight={'pad': pad, 'w_pad': w_pad, 'h_pad': h_pad})

        if self.title:
            self._fig.suptitle(self.title, fontsize=20)
            # (left, bottom, right, top) in the normalized figure coordinate that the whole subplots area
            # (including labels) will fit into
            self._fig.set_tight_layout(tight={'rect': (0, 0, 1, 0.95)})

        if save_path:
            if save_path == 'Desktop':
                if not file_name:
                    file_name = os.path.basename(inspect.stack()[-1][1])
                    file_name += options.get('append', '')
                save_path = os.path.join(os.path.expanduser('~'), 'Desktop', file_name + '.pdf')
            plt.savefig(save_path)
        else:
            with RockPy3.ignored(AttributeError):
                self._fig.canvas.manager.window.raise_()
            plt.show()


def generate_plots(n=3, xsize=5., ysize=5., columns=None, tight_layout=True):
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
        rows: tries to fit the plots in as many rows
        tight_layout: bool
            using tight_layout (True) or not (False)

    Returns
    -------
       fig matplotlib figure instance
    """
    if columns:
        b = columns
        a = np.ceil(1. * n / b).astype(int)
    else:
        a = np.floor(n ** 0.5).astype(int)
        b = np.ceil(1. * n / a).astype(int)

    fig = plt.figure(figsize=(xsize * b, ysize * a), tight_layout=tight_layout)

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
