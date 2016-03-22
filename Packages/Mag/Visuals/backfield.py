__author__ = 'mike'
import RockPy3
# import Features.backfield
import RockPy3.Packages.Mag
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.backfield
import inspect

class Backfield(Visual):
    def init_visual(self):
        self.standard_features = ['zero_lines', 'backfield_data']
        # self.standard_plt_props = {'zero_lines': {'color': 'k'}}
        self.xlabel = 'Field'
        self.ylabel = 'Moment'

    def __call__(self, *args, **kwargs):
        super(Backfield, self).__call__(*args, **kwargs)
        xlims = self.ax.get_xlim()
        self.ax.set_xlim((xlims[0], 0))

    @plot(mtypes='backfield')
    def feature_backfield_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.backfield.backfield_data(self.ax, mobj, **plt_props)

    @plot(mtypes='backfield')
    def feature_rockmag_results(self, mobj, plt_props=None):

        names = ('mrs', 'bcr')
        tex_names = dict(mrs= '$M_{rs}$', bcr= '$B_{cr}$')
        results = {
            'mrs': '${:.1e}$'.format(mobj.results['mrs'].v[0]),
            'bcr': '${:.1f}$'.format(mobj.results['bcr'].v[0] * 1000) }

        s = '\n'.join('{}: {}'.format(tex_names[res], results[res]) for res in names)
        plt_props.update({'s': s})
        plt_props.setdefault('x', 0.05)
        plt_props.setdefault('y', 0.7)
        plt_props.setdefault('transform', 'ax')
        plt_props, txt_props, kwargs = self.separate_plt_props_from_kwargs(**plt_props)
        RockPy3.Packages.Generic.Features.generic.text_x_y(ax=self.ax, **txt_props)