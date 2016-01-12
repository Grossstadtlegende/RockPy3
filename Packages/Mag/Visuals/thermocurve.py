__author__ = 'mike'
import RockPy3
# import Features.backfield
import RockPy3.Packages.Mag
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.thermocurve
import inspect

class Thermocurve(Visual):

    def init_visual(self):
        self.standard_features = ['zero_lines', 'thermocurve_data']
        self.xlabel = 'Temperature [K]'
        self.ylabel = 'Moment Am^2'

    @plot(mtypes=['thermocurve'], overwrite_mobj_plt_props={'marker':''})
    def feature_thermocurve_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.thermocurve.thermocurve_data(self.ax, mobj, **plt_props)\

    @plot(mtypes=['thermocurve'], overwrite_mobj_plt_props={'marker':''})
    def feature_thermocurve_data_colored(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.thermocurve.thermocurve_data_colored(self.ax, mobj, **plt_props)
