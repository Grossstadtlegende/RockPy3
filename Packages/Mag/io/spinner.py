__author__ = 'mike'
import numpy as np

from RockPy3.core import io
import re


class Jr6(io.ftype):
    def __init__(self, dfile, sample_name):
        super().__init__(dfile=dfile, sname=sample_name)
        self.raw_data = self.simple_import()
        self.raw_data = [i for i in self.raw_data if sample_name.lower() in i.lower()]
        self.header = ('variable', 'x', 'y', 'z')
        self.floats = ('x', 'y', 'z', 'exponent')
        self.modes = np.array([i[10:19].strip().lower() for i in self.raw_data])
        x = [float(i[19:25].strip()) for i in self.raw_data]
        y = [float(i[25:30].strip()) for i in self.raw_data]
        z = [float(i[30:36].strip()) for i in self.raw_data]
        self.exponent = np.array([10 ** float(i[37:40].strip()) for i in self.raw_data])
        self.xyz = np.c_[x * self.exponent, y * self.exponent, z * self.exponent]*1e-5

        self.data = np.array((self.variables, self.xyz[:,0], self.xyz[:,1], self.xyz[:,2])).T

    @property
    def variables(self):
        non_decimal = re.compile(r'[^\d.]+')
        out = []
        for var in self.modes:
            if var == 'NRM':
                out.append(20)
            else:
                out.append(float(non_decimal.sub('', var)))
        return out

