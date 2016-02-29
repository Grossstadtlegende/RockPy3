__author__ = 'mike'
import RockPy3
# import Features.backfield
import RockPy3.Packages.Mag
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.hysteresis
import numpy as np


class ResultSeries(Visual):
    def __init__(self, data=None, plt_index=None, fig=None, name=None, coord=None,
                 plot_groupmean=None, plot_samplemean=None, plot_samplebase=None, plot_groupbase=None,
                 plot_other=None, base_alpha=None,
                 result=None, series=None,
                 title=None,
                 **options):

        # has to be before the super call because the initialize need both result and series
        self.result = result
        self.series = series

        # so the calculation parameters can be checked for only this result
        options.setdefault('result', result)
        if not title:
            title = '{} vs. {}'.format(result, series)

        super(ResultSeries, self).__init__(
            data=data, plt_index=plt_index,
            fig=fig, name=name, coord=coord,
            plot_groupmean=plot_groupmean, plot_groupbase=plot_groupbase,
            plot_samplemean=plot_samplemean, plot_samplebase=plot_samplebase,
            plot_other=plot_other, base_alpha=base_alpha,
            title=title,
            **options
        )

    def init_visual(self):
        self.standard_features = ['result_series_data', 'result_series_errorbars']
        self.xlabel = self.series
        self.ylabel = self.result

    @plot(result_feature=True)
    def feature_result_series_data(self, data, plt_props=None):
        self.log.debug('PLOTTING WITH: {}'.format(plt_props))
        self.ax.plot(data[self.series].v, data[self.result].v, **plt_props)

    @plot(result_feature=True, overwrite_mobj_plt_props={'alpha': 1., 'marker': '', 'linestyle': ''})
    def feature_result_series_errorbars(self, data, plt_props=None):

        if all(np.isnan(i) for i in data[self.result].e):
            self.log.debug('NO error data found cant plot')
            return
        self.ax.errorbar(x=data[self.series].v, y=data[self.result].v,
                         yerr=data[self.result].e, **plt_props)

    @plot(result_feature=True)
    def feature_result_series_errorfill(self, data, plt_props=None):
        if all(np.isnan(i) for i in data[self.result].e):
            self.log.debug('NO error data found cant plot')
            return
        # take out marker - no marker in fillbetween
        plt_props.pop('marker', None)
        plt_props.pop('linestlye', None)

        plt_props.update(dict(alpha=0.1))
        self.ax.fill_between(x=data[self.series].v,
                             y1=data[self.result].v + data[self.result].e,
                             y2=data[self.result].v - data[self.result].e,
                             **plt_props)

    @plot(result_feature=True)
    def feature_result_series_errorfill_2sigma(self, data, plt_props=None):
        if all(np.isnan(i) for i in data[self.result].e):
            self.log.warning('NO error data found cant plot')
            return
        # take out marker - no marker in fillbetween
        plt_props.pop('marker', None)

        plt_props.update(dict(alpha=0.05))
        self.ax.fill_between(x=data[self.series].v,
                             y1=data[self.result].v + 2 * data[self.result].e,
                             y2=data[self.result].v - 2 * data[self.result].e,
                             **plt_props)
