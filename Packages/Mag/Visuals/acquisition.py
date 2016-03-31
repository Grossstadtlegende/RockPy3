__author__ = 'mike'
import RockPy3
# import Features.backfield
import RockPy3.Packages.Mag
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.acquisition

class Acquisition(Visual):

    def init_visual(self):
        self.standard_features = ['zero_lines', 'acquisition_data']
        self.xlabel = 'Step'
        self.ylabel = 'Moment'

    @plot(mtypes=['acquisition', 'arm_acquisition', 'parm_acquisition', 'irm_acquisition'])
    def feature_acquisition_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.acquisition.acquisition_data(self.ax, mobj, **plt_props)

    @plot(mtypes=['parm_acquisition'])
    def feature_cumulative_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.acquisition.cumsum_acquisition_data(self.ax, mobj, **plt_props)

class Parm_Acquisition(Acquisition):
   def init_visual(self):
        self.standard_features = ['zero_lines', 'grid', 'acquisition_data']
        self.xlabel = 'Step [mT]'
        self.ylabel = 'Moment'

class Irm_Acquisition(Acquisition):
    def init_visual(self):
        self.standard_features = ['zero_lines', 'grid', 'acquisition_data']
        self.xlabel = 'Field [T]'
        self.ylabel = 'Moment [$Am^2$]'

