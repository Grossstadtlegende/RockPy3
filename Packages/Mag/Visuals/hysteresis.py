__author__ = 'mike'
import RockPy3
# import Features.backfield
import RockPy3.Packages.Mag
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.hysteresis
import inspect
class Hysteresis(Visual):
    def init_visual(self):
        self.standard_features = ['zero_lines', 'grid', 'hysteresis_data']
        # self.standard_plt_props = {'zero_lines': {'color': 'k'}}
        self.xlabel = 'Field'
        self.ylabel = 'Moment'

    @plot(mtypes='hysteresis', overwrite_mobj_plt_props={'marker':''})
    def feature_hysteresis_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.hysteresis.hysteresis(self.ax, mobj, **plt_props)
        pass

    @plot(mtypes='hysteresis')
    def feature_hysteresis_error(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.hysteresis.hysteresis_error(self.ax, mobj, **plt_props)
        pass

    @plot(mtypes='hysteresis')
    def feature_virgin(self, mobj, **plt_props):
        pass

# class Fabian2003(base.Visual):
#     _required = [('hysteresis', 'backfield')]
#
#     def init_visual(self):
#         self.features = [self.feature_hysteresis_data, self.feature_backfield_data, self.feature_zero_lines]
#         # self.features = [self.feature_backfield_data, self.feature_zero_lines]
#
#         self.xlabel = 'Field'
#         self.ylabel = 'Moment'
#
#     @feature(mtypes='hysteresis')
#     def feature_hysteresis_data(self, **plt_props):
#         Features.hysteresis.hysteresis(self.ax, plt_props.pop('mobj'), **plt_props)
#
#     @feature(mtypes='backfield')
#     def feature_backfield_data(self, **plt_props):
#         plt_props.get('mobj').marker = ''
#         Features.backfield.backfield(self.ax, plt_props.pop('mobj'), **plt_props)
#
#     @feature(plt_frequency='single')
#     def feature_zero_lines(self, mobj=None, **plt_props):
#         color = plt_props.pop('color', 'k')
#         zorder = plt_props.pop('zorder', 0)
#
#         self.ax.axhline(0, color=color, zorder=zorder,
#                         **plt_props)
#         self.ax.axvline(0, color=color, zorder=zorder,
#                         **plt_props)
