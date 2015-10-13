__author__ = 'mike'
from RockPy3.Packages.Generic.Measurements import *

import os
from os.path import join

import glob
modules = glob.glob(os.path.dirname(__file__)+"/*.py")
__all__ = [ os.path.basename(f)[:-3] for f in modules]

# test_data_path = join(os.getcwd().split('RockPy')[0], 'RockPy', 'Packages', 'Generic', 'testdata')