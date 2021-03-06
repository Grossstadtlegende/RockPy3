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
                 data=None,
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

        self.data = RockPy3.core.utils.sort_input(data)

        mlist, mean_list = RockPy3.core.utils.mlist_from_input(data)

        self.calculation_parameter = []

        self.plot_groupmean, self.plot_samplemean = plot_groupmean, plot_samplemean
        self.plot_groupbase, self.plot_samplebase = plot_groupbase, plot_samplebase
        self.plot_other = plot_other

        self.result_from_means = result_from_means
        self.base_alpha = base_alpha
        self.ignore_samples = ignore_samples

    def add_data(self, data, groupmean=True, samplemean=True, base=True, other=True):
        self.data = RockPy3.core.utils.add_to_input(data=data, to_add_to=self.data,
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

    def add_visual(self, visual, name=None, data=None,
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
        visuals = RockPy3._to_tuple(visual)
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
                    data=data, plt_index=n, fig=self, name=name,
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
                x = len(v.implemented_features().keys())-1
                implemented = sorted(v.implemented_features().keys())
                for line in range(x%4):
                    if line == 0:
                        self.log.info('\t{:20}:\t{}'.format(n.upper(), ', '.join(implemented[:4])))
                    else:
                        end = (line+1)*4 if (line+1)*4 < x else x
                        self.log.info('\t{:20} \t{}'.format('', ', '.join(implemented[line*4:end])))
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
             xlim=None, ylim=None,
             equal_lims=False, center_lims=False,
             save_path=None,
             pad=0.4, w_pad=0.5, h_pad=1.0,
             file_name=None, format='pdf',
             legend=True, sort_labels = True,
             return_figure=False,
             append_to_filename = '',
             ):
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
            if visual.xlim:
                visual.ax.set_xlim(visual.xlim)
            if visual.ylim:
                visual.ax.set_ylim(visual.ylim)

            if visual.xscale:
                visual.ax.set_xscale(visual.xscale)
            if visual.yscale:
                visual.ax.set_yscale(visual.yscale)

            # prevent scientific notation for x axis
            # if type in ('thermocurve', ):
            visual.ax.ticklabel_format(style='plain', axis='x')
            # else:
            #     xlim = visual.ax.get_xlim()
            #     ylim = visual.ax.get_ylim()

        if xlim == 'equal' or ylim == 'equal' or equal_lims:
            if equal_lims:
                xlim, ylim = self.get_xylims()

            if center_lims:
                xl = max(np.abs(xlim))
                yl = max(np.abs(ylim))
                xlim = [-xl, xl]
                ylim = [-yl, yl]

            # cycle through visuals to set
            for name, type, visual in self._visuals:
                if xlim == 'equal' or equal_lims:
                    visual.ax.set_xlim(xlim)
                if ylim == 'equal' or equal_lims:
                    visual.ax.set_ylim(ylim)

        # prepare legends for individual visuals
        for name, type, visual in self._visuals:
            # check if the legend should be drawn accoring to the visual.legend dictionary
            if not visual.show_legend():
                continue

            if not legend:
                break
            handles, labels = visual.ax.get_legend_handles_labels()
            if not all(i for i in (handles, labels)):
                continue

            # sorting of labels
            if sort_labels:
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            visual.ax.legend(handles, labels, **visual.legend_options)

        # check if two entries and each is float or int
        if xlim:
            if len(xlim) == 2 and any(isinstance(i, (float, int)) for i in xlim):
                for name, type, visual in self._visuals:
                    visual.ax.set_xlim(xlim)
        # check if two entries and each is float or int
        if ylim:
            if len(ylim) == 2 and any(isinstance(i, (float, int)) for i in ylim):
                for name, type, visual in self._visuals:
                    visual.ax.set_ylim(ylim)

        self._fig.set_tight_layout(tight={'pad': pad, 'w_pad': w_pad, 'h_pad': h_pad})

        if self.title:
            self._fig.suptitle(self.title, fontsize=20)
            # (left, bottom, right, top) in the normalized figure coordinate that the whole subplots area
            # (including labels) will fit into
            self._fig.set_tight_layout(tight={'rect': (0, 0, 1, 0.95)})

        if return_figure:
            return self._fig

        if save_path:
            if save_path.lower() == 'desktop':
                if not file_name:
                    file_name = os.path.basename(inspect.stack()[-1][1])
                    file_name += append_to_filename
                save_path = os.path.join(os.path.expanduser('~'), 'Desktop', file_name)
            else:
                save_path = os.path.join(save_path, file_name)

            if not format in file_name:
                save_path +='.'
                save_path += format

            plt.savefig(save_path)
        else:
            with RockPy3.ignored(AttributeError):
                self._fig.canvas.manager.window.raise_()
            plt.show()
            plt.close('all')


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

    fig = plt.figure(figsize=(xsize * b, ysize * a))

    axes = []

    for i in range(1, n + 1):
        ax1 = fig.add_subplot(a, b, i)
        for ax in [ax1]:
            ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
            ax.xaxis.major.formatter._useMathText = True
            ax.yaxis.major.formatter._useMathText = True
        axes.append([ax1, None, None])

    if tight_layout:
        fig.set_tight_layout(tight_layout)
    return fig, axes
