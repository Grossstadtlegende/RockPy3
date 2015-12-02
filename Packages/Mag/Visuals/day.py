__author__ = 'mike'
import RockPy3
# import Features.day
import RockPy3.Packages.Mag
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.day
import inspect


class Day(Visual):
    def init_visual(self):
        self.standard_features = ['zero_lines', 'day_grid', 'day_data']
        # self.standard_plt_props = {'zero_lines': {'color': 'k'}}
        self.xlabel = 'Field'
        self.ylabel = 'Moment'

    @plot(mtypes=('hysteresis', 'backfield'))
    def feature_day_data(self, mobj, plt_props=None):
        # print(mobj)
        RockPy3.Packages.Mag.Features.day.day_data(self.ax, mobj, **plt_props)

    @plot(single=True)
    def feature_day_grid(self, plt_props=None):
        RockPy3.Packages.Mag.Features.day.day_grid(ax=self.ax)

    @plot(single=True)
    def feature_sd_md_mixline_1(self, plt_props=None):
        # after Dunlup2002a
        plt_props['color'] = 'k'
        plt_props['alpha'] = 0.7
        RockPy3.Packages.Mag.Features.day.sd_md_mixline_1(ax=self.ax, **plt_props)

    @plot(single=True)
    def feature_sd_md_mixline_2(self, plt_props=None):
        # after Dunlup2002a
        plt_props['color'] = 'k'
        plt_props['alpha'] = 0.7
        RockPy3.Packages.Mag.Features.day.sd_md_mixline_2(ax=self.ax, **plt_props)

    @plot(single=True)
    def feature_sp_envelope(self, plt_props=None):
        # after Dunlup2002a
        plt_props['color'] = 'k'
        plt_props['alpha'] = 0.7
        RockPy3.Packages.Mag.Features.day.sp_envelope(ax=self.ax, **plt_props)

    @plot(single=True)
    def feature_sd_sp_10nm(self, plt_props=None):
        # after Dunlup2002a
        plt_props['color'] = 'k'
        plt_props['alpha'] = 0.7
        RockPy3.Packages.Mag.Features.day.sd_sp_10nm(ax=self.ax, **plt_props)

    @plot(single=True)
    def feature_errorbars(self, plt_props=None):
        # after Dunlup2002a
        plt_props['color'] = 'k'
        plt_props['alpha'] = 0.7
        RockPy3.Packages.Mag.Features.day.errorbars(ax=self.ax, mobj=plt_props.pop('mobj'), **plt_props)


if __name__ == '__main__':
    Study = RockPy3.Study
    s = Study.add_sample(name='S1')
    hys = s.add_measurement(mtype='hysteresis', fpath='/Users/Mike/GitHub/RockPy_presentation/hys.001', ftype='vsm')
    coe = s.add_measurement(mtype='backfield', fpath='/Users/Mike/GitHub/RockPy_presentation/coe.001', ftype='vsm')

    fig = RockPy3.Figure(fig_input=Study)
    v = fig.add_visual(visual='day', color='r', markersize=10, xlims=[1, 6])
    # print(v.xlims)
    v.add_feature(feature='sd_md_mixline_1', marker='')
    fig.show()
