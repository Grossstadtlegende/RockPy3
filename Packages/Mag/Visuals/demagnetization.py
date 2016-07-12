__author__ = 'mike'
import RockPy3
# import Features.backfield
import RockPy3.Packages.Mag
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.demagneitzation
import RockPy3.Packages.Mag.Features.acquisition
import inspect


class Demagnetization(Visual):
    def init_visual(self):
        self.standard_features = ['zero_lines', 'demagnetization_data']
        self.xlabel = 'Step'
        self.ylabel = 'Moment'

    @plot(mtypes=['demagnetization', 'afdemagnetization'])
    def feature_demagnetization_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.demagneitzation.demagnetization_data(self.ax, mobj, **plt_props)

    @plot(mtypes=['acquisition', 'parm_acquisition'])
    def feature_cumulative_acquisition_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.acquisition.cumsum_acquisition_data(self.ax, mobj, **plt_props)


    @plot(mtypes=['demagnetization', 'afdemagnetization'])
    def feature_demagnetization_components(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.demagneitzation.demagnetization_data(self.ax, mobj, dtype='x', **plt_props)
        RockPy3.Packages.Mag.Features.demagneitzation.demagnetization_data(self.ax, mobj, dtype='y', **plt_props)
        RockPy3.Packages.Mag.Features.demagneitzation.demagnetization_data(self.ax, mobj, dtype='z', **plt_props)

class Af_Demagnetization(Demagnetization):
    def init_visual(self):
        self.standard_features = ['zero_lines', 'grid', 'demagnetization_data']
        self.xlabel = 'Step [mT]'
        self.ylabel = 'Moment'
