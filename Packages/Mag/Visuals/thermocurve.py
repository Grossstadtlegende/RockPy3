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
        self.xlabel = 'Temperatures'
        self.ylabel = 'Moment $Am^2$'

    @plot(mtypes=['thermocurve'], overwrite_mobj_plt_props={'marker':''})
    def feature_thermocurve_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.thermocurve.thermocurve_data(self.ax, mobj, **plt_props)\

    @plot(mtypes=['thermocurve'], overwrite_mobj_plt_props={'marker':''})
    def feature_thermocurve_data_colored(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.thermocurve.thermocurve_data_colored(self.ax, mobj, **plt_props)


    @plot(mtypes=['thermocurve'], overwrite_mobj_plt_props={'marker': ''})
    def feature_thermocurve_derivative(self, mobj, twinx=True, plt_props=None):
        RockPy3.Packages.Mag.Features.thermocurve.thermocurve_derivative(self.ax, mobj, **plt_props)

if __name__ == '__main__':
    vftb_rmp = '/Users/Mike/Dropbox/experimental_data/pyrrhotite/VFTB/msm17591final.rmp'
    s2 = RockPy3.Study.add_sample(name='MSM17591')
    m = s2.add_measurement(mtype='rmp', ftype='vftb', fpath=vftb_rmp)
    f = RockPy3.Figure(data=s2, figsize=(6, 3))
    v = f.add_visual('thermocurve')
    v.add_feature('thermocurve_derivative', twinx=True)
    v.title = s2.name
    f.show(xlim=(20, 350))