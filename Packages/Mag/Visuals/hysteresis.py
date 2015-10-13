__author__ = 'mike'
import RockPy3
# import Features.backfield
# import Features.hysteresis
from RockPy3.core.utils import feature


class Hysteresis(RockPy3.core.visual.Visual):
    # _required for searching through samples for plotables
    _required = [('hysteresis',), ]

    def init_visual(self):
        self.features = [self.feature_zero_lines, self.feature_hysteresis_data]

        self.xlabel = 'Field'
        self.ylabel = 'Moment'

    @feature(mtypes='hysteresis')
    def feature_hysteresis_data(self, **plt_opt):
        # Features.hysteresis.hysteresis(self.ax, plt_opt.pop('mobj'), **plt_opt)
        pass

    @feature(mtypes='hysteresis')
    def feature_virgin(self, mobj, **plt_opt):
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
#     def feature_hysteresis_data(self, **plt_opt):
#         Features.hysteresis.hysteresis(self.ax, plt_opt.pop('mobj'), **plt_opt)
#
#     @feature(mtypes='backfield')
#     def feature_backfield_data(self, **plt_opt):
#         plt_opt.get('mobj').marker = ''
#         Features.backfield.backfield(self.ax, plt_opt.pop('mobj'), **plt_opt)
#
#     @feature(plt_frequency='single')
#     def feature_zero_lines(self, mobj=None, **plt_opt):
#         color = plt_opt.pop('color', 'k')
#         zorder = plt_opt.pop('zorder', 0)
#
#         self.ax.axhline(0, color=color, zorder=zorder,
#                         **plt_opt)
#         self.ax.axvline(0, color=color, zorder=zorder,
#                         **plt_opt)
