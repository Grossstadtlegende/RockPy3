__author__ = 'mike'
from time import clock
import RockPy3
import numpy as np

from RockPy3.core import io
from os.path import join
from copy import deepcopy

class SushiBar(io.ftype):
    def __init__(self, dfile, dialect=None):
        super(SushiBar, self).__init__(dfile=dfile, dialect=dialect)
