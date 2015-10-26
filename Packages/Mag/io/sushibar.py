__author__ = 'mike'
from time import clock
import RockPy3
import numpy as np

from RockPy3.core import io
from os.path import join
from copy import deepcopy
from RockPy3.core.utils import convert_time
class SushiBar(io.ftype):
    def __init__(self, dfile, dialect=None):
        super(SushiBar, self).__init__(dfile=dfile, dialect=dialect)
        data = [i.split('\t') for i in self.simple_import()]
        self.header = ['meas. time', 'x', 'y', 'z', 'M', 'a95', 'sM',
                  'npos', 'par1', 'par2', 'par3', 'par4', 'par5', 'par6']
        header = data.pop(0)
        self.raw_data = {}
        for d in data:
            self.raw_data.setdefault(d[0], [])
            try:
                aux = [convert_time(d[header.index('meas. time')])]
                aux.extend([d[header.index(h)] for h in self.header[1:]])
                aux = [i if i != 'None' else np.nan for i in aux]
                self.raw_data[d[0]].append(aux)
            except ValueError:
                pass

        for sample in self.raw_data:
            self.raw_data[sample] = np.array(self.raw_data[sample]).astype(float)