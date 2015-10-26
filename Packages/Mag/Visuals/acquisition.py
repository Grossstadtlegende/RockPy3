__author__ = 'mike'
import RockPy3
# import Features.backfield
import RockPy3.Packages.Mag
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.acquisition
import inspect

class Acquisition(Visual):

    def init_visual(self):
        self.standard_features = ['zero_lines', 'grid', 'acquisition_data']
        self.xlabel = 'Step'
        self.ylabel = 'Moment'

    @plot(mtypes=['acquisition', 'arm_acquisition'])
    def feature_acquisition_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.acquisition.acquisition_data(self.ax, mobj, **plt_props)\

    @plot(mtypes=['acquisition', 'arm_acquisition'])
    def feature_cumulative_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.acquisition.cumsum_acquisition_data(self.ax, mobj, **plt_props)

class Arm_acqisition(Acquisition):
   def init_visual(self):
        self.standard_features = ['zero_lines', 'grid', 'acquisition_data']
        self.xlabel = 'Step [mT]'
        self.ylabel = 'Moment'