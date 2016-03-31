__author__ = 'mike'
import RockPy3
# import Features.backfield
import RockPy3.Packages.Mag
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.henkel

class Henkel(Visual):
    def init_visual(self):
        self.standard_features = ['henkel_data', 'one_to_onle_line']
        self.xlabel = 'backfield moment ($Am^2$)'
        self.ylabel = 'IRM moment ($Am^2$)'


    def __call__(self, *args, **kwargs):
        super(Henkel, self).__call__(*args, **kwargs)
        self.ax.set_xlim((-1, 1))
        self.ax.set_ylim((0, 1))

    @plot(mtypes=[('backfield', 'irm_acquisition')])
    def feature_henkel_data(self, mobj, reference='coe', plt_props=None):
        RockPy3.Packages.Mag.Features.henkel.henkel_data(self.ax, mobj=mobj, reference=reference, **plt_props)

    @plot(single=True)
    def feature_one_to_onle_line(self, plt_props=None):
        RockPy3.Packages.Mag.Features.henkel.one_to_one_line(self.ax, **plt_props)
