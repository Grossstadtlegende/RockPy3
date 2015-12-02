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

    @plot(mtypes='backfield')
    def feature_backfield_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.backfield.backfield_data(self.ax, mobj, **plt_props)

