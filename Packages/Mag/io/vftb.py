__author__ = 'mike'
from time import clock
import RockPy3
import numpy as np

from RockPy3.core import io
from os.path import join
from copy import deepcopy


class Vftb(io.ftype):
    def __init__(self, dfile, dialect=None):
        super(Vftb, self).__init__(dfile, dialect)
        self.raw_data = [i.strip('\r\n').split('\t') for i in open(self.file_name).readlines() if not '#' in i]
        self.mass = float(self.raw_data[0][1].split()[1]) * 1e-6 #mass in 'kg'
        # finding set indices
        idx = [j for j, v2 in enumerate(self.raw_data) for i, v in enumerate(v2) if v.startswith('Set')]
        # appending length of measurement (-1 because of empty last line
        idx.append(len(self.raw_data))
        self.set_idx = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]
        self.units = self.get_units()
        self.header = self.get_header()
        self.data = self.get_data()

    def get_header(self):
        header = self.raw_data[self.set_idx[0][0] + 1]
        header = ['_'.join(i.split(' / ')[0].split()) for i in header]  # only getting the column name without the unit
        return header

    def get_units(self):
        units = self.raw_data[self.set_idx[0][0] + 1]
        units = ['/'.join(i.split(' / ')[1:]) for i in units]  # only getting the column name without the unit
        units = [i if not 'centigrade' in i else 'C' for i in units]
        units = [i if not '%' in i else '' for i in units]
        return units

    def get_data(self):
        """
        Formats data into usable format. Using set indices for multiple measurements
        :return:
        """
        # get a list of data for each index
        data = np.array([np.array(self.raw_data[i[0] + 2:i[1]]) for i in self.set_idx])
        data = np.array([np.array([j for j in i if len(j) >1]) for i in data ])
        # print data[0]
        data = map(self.replace_na, data)
        data = np.array([i.astype(float) for i in data])
        data = self.convert_to_T(data)

        # change unit emu/g -> Am^2/kg
        self.units =[i if not 'emu' in i else 'A m^2' for i in self.units]
        # convert to A m^2 instead of A m^2/kg
        convert = [self.mass if 'g' in v else 1 for i,v in enumerate(self.units)]
        data = np.array([i * convert for i in data])

        # some vftb files have a prefix of E-3
        # -> data is corrected
        convert = [1e-3 if 'E-3' in v else 1 for i,v in enumerate(self.units)]
        data = np.array([i * convert for i in data])
        return data

    def replace_na(self, data):
        out = [['0.' if j == 'n/a' else j for j in i] for i in data]
        out = np.array(out)
        return out

    def convert_to_T(self, data):
        for i in data:
            i[:, 0] /= 10000
        self.units[0] = 'T'
        return data


if __name__ == '__main__':
    vftb_file = '/Users/Mike/Dropbox/experimental_data/001_PintP/LF4C/VFTB/P0-postTT/140310_1a.hys'
    print(Vftb(dfile=vftb_file).data)