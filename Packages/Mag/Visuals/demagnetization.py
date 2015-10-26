__author__ = 'mike'
import RockPy3
# import Features.backfield
import RockPy3.Packages.Mag
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.demagneitzation
import inspect

class Demagnetization(Visual):
    def init_visual(self):
        self.standard_features = ['zero_lines', 'grid', 'demagnetization_data']
        self.xlabel = 'Step'
        self.ylabel = 'Moment'

    @plot(mtypes=['demagnetization', 'afdemagnetization'])
    def feature_demagnetization_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.demagneitzation.demagnetization_data(self.ax, mobj, **plt_props)
        pass