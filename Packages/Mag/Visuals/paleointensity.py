__author__ = 'mike'
import RockPy3
# import Features.backfield
import RockPy3.Packages.Mag
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.paleointensity
import RockPy3.Packages.Mag.Features.acquisition
import RockPy3.Packages.Generic.Features.generic
import inspect
import numpy as np


class Paleointensity(Visual):
    def init_visual(self):
        self.standard_features = ['zero_lines', 'paleointensity_data', 'least_square_line']
        self.xlabel = 'Step'
        self.ylabel = 'Moment'

    @plot(mtypes=['paleointensity'])
    def feature_paleointensity_data(self, mobj, plt_props=None):
        RockPy3.Packages.Mag.Features.paleointensity.paleointensity_data_points(self.ax, mobj, **plt_props)

    @plot(mtypes=['paleointensity'], overwrite_mobj_plt_props=dict(marker=''))
    def feature_least_square_line(self, mobj, plt_props=None):

        # get the slope result
        slope = mobj.result_slope(**self.calculation_parameter)
        # get the y-intercept result
        intercept = mobj.result_y_int(**self.calculation_parameter)
        # get the calculation limits
        vars = [mobj.calculation_parameter['slope']['var_min'], mobj.calculation_parameter['slope']['var_max']]

        idx = [i for i, v in enumerate(mobj.data['acquisition']['variable'].v) if v in vars]
        x = mobj.data['acquisition'].filter_idx(idx)[mobj.calculation_parameter['slope']['component']].v
        y = intercept[0] + slope[0] * x
        RockPy3.Packages.Generic.Features.generic.plot_x_y(ax=self.ax, xdata=x, ydata=y, **plt_props)


class PseudoThellier(Paleointensity):
    def init_visual(self):
        self.standard_features = ['zero_lines', 'paleointensity_data', 'least_square_line']
        self.xlabel = 'ARM acquired'
        self.ylabel = 'NRM remaining'


if __name__ == '__main__':
    step1C = '/Users/mike/Dropbox/experimental_data/RelPint/Step1C/1c.csv'
    step1B = '/Users/mike/Dropbox/experimental_data/RelPint/Step1B/IG_1291A.cmag.xml'

    S = RockPy3.Study
    s = S.add_sample(name='IG_1291A')
    pARM_acq = s.add_measurement(mtype='parmacq', fpath=step1C, ftype='sushibar', series=[('ARM', 50, 'muT')])
    NRM_AF = s.add_measurement(mtype='afdemag', fpath=step1B, ftype='cryo', series=[('NRM', 0, '')])

    palint = s.add_measurement(mtype='paleointensity', mobj=(pARM_acq, NRM_AF))
    fig = RockPy3.Figure()
    fig.add_visual(visual='acquisition', data=pARM_acq)
    fig.add_visual(visual='AF_DEMAGNETIZATION'.lower(), data=NRM_AF, var_min=30, var_max=60)
    v = fig.add_visual(visual='pseudothellier', data=palint, var_min=30, var_max=60)
    v.add_feature(feature='least_square_line')
    fig.show()
