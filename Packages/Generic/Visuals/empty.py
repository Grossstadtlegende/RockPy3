__author__ = 'mike'
import RockPy3
import RockPy3.Packages.Mag
import RockPy3.Packages.Mag.Visuals.backfield
from RockPy3.core.utils import plot
from RockPy3.core.visual import Visual
import RockPy3.Packages.Mag.Features.hysteresis
import RockPy3.Packages.Mag.Features.backfield
from collections import OrderedDict
import numpy as np

class Empty(Visual):
    def init_visual(self):
        self.standard_features = ['zero_lines']
        self.xlabel = ''
        self.ylabel = ''
