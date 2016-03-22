__author__ = 'mike'
import RockPy3
import RockPy3.Packages.Mag
import RockPy3.Packages.Mag.Visuals.backfield
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.hysteresis
import RockPy3.Packages.Mag.Features.backfield
from collections import OrderedDict
import numpy as np

class Hysteresis(Visual):
    def init_visual(self):
        self.standard_features = ['zero_lines', 'hysteresis_data']
        self.xlabel = 'Field [$T$]'
        self.ylabel = 'Moment [$Am^2$]'

    def __call__(self, *args, **kwargs):
        super(Hysteresis, self).__call__(*args, **kwargs)
        xlims = max(abs(d) for l in self.ax.lines for d in l.get_data()[1])
        self.ax.set_xlim((-xlims, xlims))

    @plot(mtypes='hysteresis', overwrite_mobj_plt_props={'marker': ''})
    def feature_hysteresis_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.hysteresis.hysteresis(self.ax, mobj, **plt_props)

    @plot(mtypes='hysteresis')
    def feature_hysteresis_error(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.hysteresis.hysteresis_error(self.ax, mobj, **plt_props)

    @plot(mtypes='hysteresis')
    def feature_virgin(self, mobj, **plt_props):
        pass

    @plot(mtypes='hysteresis', overwrite_mobj_plt_props={'marker': ''})
    def feature_reversible_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.hysteresis.irreversible(self.ax, mobj, **plt_props)

    @plot(mtypes='hysteresis', overwrite_mobj_plt_props={'marker': ''})
    def feature_irreversible_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.hysteresis.reversible(self.ax, mobj, **plt_props)

    @plot(mtypes='hysteresis', overwrite_mobj_plt_props={'marker': ''})
    def feature_hysteresis_derivative_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.hysteresis.hysteresis_derivative(self.ax, mobj, **plt_props)

    @plot(mtypes='hysteresis')
    def feature_rockmag_results(self, mobj, plt_props=None):

        names = ('ms', 'mrs', 'mrs_ms', 'bc')
        tex_names = dict(ms= '$M_s$', mrs= '$M_{rs}$', mrs_ms =  '$M_{rs}/M_{s}$', bc = '$B_c$')
        results = {
            'ms': '${:.2e}$'.format(mobj.results['ms'].v[0]),
            'mrs': '${:.1e}$'.format(mobj.results['mrs'].v[0]),
            'mrs_ms': '${:.2f}$'.format(mobj.results['mrs_ms'].v[0]),
            'bc': '${:.1f}$'.format(mobj.results['bc'].v[0] * 1000),
        }

        s = '\n'.join('{}: {}'.format(tex_names[res], results[res]) for res in names)
        plt_props.update({'s': s})
        plt_props.setdefault('x', 0.55)
        plt_props.setdefault('y', 0.05)
        plt_props.setdefault('transform', 'ax')
        plt_props, txt_props, kwargs = self.separate_plt_props_from_kwargs(**plt_props)
        RockPy3.Packages.Generic.Features.generic.text_x_y(ax=self.ax, **txt_props)

    @plot(mtypes='backfield')
    def feature_backfield_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.backfield.backfield_data(self.ax, mobj, **plt_props)


class Fabian2003(Hysteresis, RockPy3.Packages.Mag.Visuals.backfield.Backfield):
    def init_visual(self):
        self.standard_features = ['hysteresis_data', 'backfield_data', 'zero_lines']
        self.xlabel = 'Field'
        self.ylabel = 'Moment'
