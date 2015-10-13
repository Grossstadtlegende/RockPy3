__author__ = 'mike'
from RockPy3.Packages.Mag.Measurements import *
from RockPy3.Packages.Mag.io import *
from RockPy3.Packages.Mag.Visuals import *
# from io import *

import os
from os.path import join

import glob
modules = glob.glob(os.path.dirname(__file__)+"/*.py")
__all__ = [ os.path.basename(f)[:-3] for f in modules]

test_data_path = join(os.getcwd().split('RockPy')[0], 'RockPy', 'Packages', 'Mag', 'testdata')