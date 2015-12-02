__author__ = 'mike'
import RockPy3
# import Features.backfield
import RockPy3.Packages.Mag
import RockPy3.Packages.Mag.Visuals.backfield

from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.hysteresis
import inspect
class Hysteresis(Visual):
    def init_visual(self):
        self.standard_features = ['zero_lines', 'hysteresis_data']
        # self.standard_plt_props = {'zero_lines': {'color': 'k'}}
        self.xlabel = 'Field'
        self.ylabel = 'Moment'

    @plot(mtypes='hysteresis', overwrite_mobj_plt_props={'marker':''})
    def feature_hysteresis_data(self, mobj, plt_props=None):
        try:
            RockPy3.Packages.Mag.Features.hysteresis.hysteresis(self.ax, mobj, **plt_props)
        except TypeError:
            pass

    @plot(mtypes='hysteresis')
    def feature_hysteresis_error(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.hysteresis.hysteresis_error(self.ax, mobj, **plt_props)

    @plot(mtypes='hysteresis')
    def feature_virgin(self, mobj, **plt_props):
        pass

    @plot(mtypes='hysteresis', overwrite_mobj_plt_props={'marker':''})
    def feature_reversible_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.hysteresis.irreversible(self.ax, mobj, **plt_props)

    @plot(mtypes='hysteresis', overwrite_mobj_plt_props={'marker':''})
    def feature_irreversible_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.hysteresis.reversible(self.ax, mobj, **plt_props)

class Fabian2003(Hysteresis, RockPy3.Packages.Mag.Visuals.backfield.Backfield):
    def init_visual(self):
        self.standard_features = ['feature_hysteresis_data', 'feature_backfield_data', 'feature_zero_lines']
        self.xlabel = 'Field'
        self.ylabel = 'Moment'