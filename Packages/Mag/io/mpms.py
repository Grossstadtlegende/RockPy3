__author__ = 'Mike'
from RockPy3.core import io
from RockPy3.core.utils import convert_time
import numpy as np


class Mpms(io.ftype):
    def __init__(self, dfile, dialect=None):
        self.file_name = dfile
        self.raw_data = self.simple_import()
        self.raw_data = [i for i in self.raw_data if i]
        self.info = [i.split(',') for i in self.raw_data if i.startswith('INFO')]
        self.info = {i[1].lower().strip().replace(':', ''): i[2].strip() for i in self.info if len(i) > 2}
        self.name = self.info.pop('name', None)
        self.data_idx = [i + 1 for i, v in enumerate(self.raw_data) if v.startswith('[Data]')][0]
        self.header = self.raw_data[self.data_idx].split(',')
        self.units = [i.split('(')[1].replace(')', '') if len(i.split('(')) > 1 else '' for i in self.header]

        # get rid of units in header
        self.header = [i.split('(')[0].strip() for i in self.raw_data[self.data_idx].split(',')]
        self.data = np.array(
            [[n if n else np.nan for n in i.split(',')] for i in self.raw_data[self.data_idx + 1:]]).astype(float)

        # get rid of nan columns
        nan_idx = [i for i, v in enumerate(self.data[0]) if np.isnan(v)]
        self.data = np.delete(self.data, nan_idx, axis=1)
        self.units = [v for i, v in enumerate(self.units) if i not in nan_idx]
        self.header = [v for i, v in enumerate(self.header) if i not in nan_idx]

        # convert emu -> Am2, Oe -> T ...
        for i, unit in enumerate(self.units):
            if unit == 'emu':
                self.data[:, i] *= 1e-3
                self.units[i] = 'A m^2'
            if unit == 'Oe':
                self.data[:, i] *= 1e-4
                self.units[i] = 'T'
            if unit == 'cm':
                self.data[:, i] *= 1e-2
                self.units[i] = 'm'


if __name__ == '__main__':
    # file = '/Users/Mike/Dropbox/XXXOLD_BACKUPS/__PHD/__Projects/002 Hematite Nanoparticles, Morin Transition/04 data/MPMS/S3M2/S3M2_IRM7T_0T_60_300K_Cooling.rso.dat'
    hys = '/Users/mike/Dropbox/Uni/Master Thesis/MPMS/S3M2/S3M2_Hys_7T_20K.rso.dat'
    Mpms(dfile=hys)
