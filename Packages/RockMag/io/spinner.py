__author__ = 'mike'
import numpy as np

from RockPy.core import io


class Jr6(io.ftype):
    def __init__(self, dfile, sample_name):

        super(Jr6, self).__init__(dfile=dfile, sample_name=sample_name)
        self.header = ['sample', 'mode', 'x', 'y', 'z', 'exponent']
        self.floats = ['x', 'y', 'z', 'exponent']
        self.raw_data = [i.strip('\r\n') for i in open(self.file_name).readlines()]
        self.names = [i[:10].strip() for i in self.raw_data]
        self.modes = np.array([i[10:19].strip().lower() for i in self.raw_data])
        x = [float(i[19:25].strip()) for i in self.raw_data]
        y = [float(i[25:30].strip()) for i in self.raw_data]
        z = [float(i[30:36].strip()) for i in self.raw_data]
        self.exponent = np.array([10 ** float(i[37:40].strip()) for i in self.raw_data])
        self.xyz = np.c_[x * self.exponent, y * self.exponent, z * self.exponent]
        self.sample_index = [i for i, v in enumerate(self.names) if v.lower() == sample_name.lower()]

    def get_header(self):
        return ['x', 'y', 'z']

    def get_exponent(self):
        return self.exponent[self.sample_index]

    def get_data(self):
        data = self.xyz[self.sample_index]
        out = data * 1e-5
        return out

    def get_type(self):
        return self.type[self.sample_datra]

    def _check_data_exists(self):
        if len(self.sample_index) > 0:
            return True
        else:
            return False

    def out_afdemag(self):
        fields = self.modes[self.sample_index]
        for i, v in enumerate(fields):
            if v == 'NRM':
                fields[i] = 0.
            if 'A' in v:
                fields[i] = v[1:].strip('-')
        fields = np.array(fields).astype(float)
        out = np.c_[fields, self.get_data()[:, 0], self.get_data()[:, 1], self.get_data()[:, 2]]
        return out
