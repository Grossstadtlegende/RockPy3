__author__ = 'mike'
from RockPy.core.io import ftype
import RockPy
# from RockPy.Packages import PaleoMag
import os
from RockPy.utils.general import DIL2XYZ
import time
import numpy as np

class tdt(ftype):
    idx_counter = 0
    def __init__(self, dfile, volume=6.22e-7, sample_name=None):
        """

        Parameters
        ----------
            dfile: str
                the full path to the file
            volume: float
                default: 6.11E-7 m3 the volume of a sample with 6mm diameter and 5.4mm height.
                volume of the sample in m3. Should be specified otherwise normalization is difficult
        """
        super(tdt, self).__init__(dfile=dfile, sample_name=sample_name, dialect=None)
        self.volume = volume
        self.lab_field = 35.0  # initialize standard field 35 micro Tesla
        self.geographic_coord = None
        self.bedding = None

        self.raw_data = self.simple_import()

    def extract_labfield_header(self, data):
        # check if second line in tdt file is header with bedding and labfield
        first_column = [i[0] for i in data]

        if any(first_column[0] != i for i in first_column[1:]):
            self.lab_field = float(data[0][0])
            self.geographic_coord = (float(data[0][1]), float(data[0][2]))
            self.bedding = (float(data[0][3]), float(data[0][4]))

            # remove the labfield line
            data.pop(0)

        return data

    def data(self, dtype='unnormalized_data', volume=None):
        data = getattr(self, dtype)(volume=volume)
        out = {}
        for i in data:
            out.setdefault(i[1].lower(), list()).append(i[2:])


        out.setdefault('nrm', out['th'][0])
        out['pt'].insert(0, out['th'][0])

        for i in out.keys():
            out[i] = np.array(out[i])
        return out

    @property
    def header(self):
        return ['temp', 'x', 'y', 'z', 'time']

    def formatted_data(self, volume=None):
        """
        Gets data from raw data

        column 1 contains the sample name (maximum length of 16 characters)
        column 2 the temperature (in C) and type of measurement
        column 3 the intensity (in mA/m)
        column 4 the declination
        column 5 the inclination in core coordinates

        The decimal digits of the temperature value (column 2) indicate the type of measurement:

        .00 (or .0) TH stands for thermal demagnetization
        .11 (or .1) PT denotes pTRM* acquisition
        .12 (or .2) CK defines the pTRM*-check
        .13 (or .3) TR repeated demagnetization steps
        .14 (or .4) AC indicates additivity checks.
        """
        data = [i.split('\t') for i in self.raw_data[1:]]

        data = self.extract_labfield_header(data)

        data = map(self.convert_dil, data)
        data = map(self.convert_mAm, data)
        data = map(self.determine_step_type, data)
        data = map(self.append_index, data)
        tdt.idx_counter = 0 #reset the index counter
        return data

    @staticmethod
    def convert_mAm(line):
        return line[:2]+[i/1e3 for i in line[2:]]

    def unnormalized_data(self, volume=None):
        """
        Returns the unnormalized data.
        """
        if not volume:
            volume = self.volume

        data = self.formatted_data()

        def unnormalize(line, volume):
            return line[:3] + [volume * i for i in line[3:]]

        return [unnormalize(line=i, volume=volume) for i in data]

    @staticmethod
    def append_index(line):
        tdt.idx_counter += 1
        line = line + [tdt.idx_counter]
        return line

    @staticmethod
    def determine_step_type(line):
        lookup = {'0': 'TH',
                  '1': 'PT', '11': 'PT',
                  '2': 'CK', '22': 'CK',
                  '3': 'TR', '33': 'TR',
                  '4': 'AC', '44': 'AC'}

        temp, type = ('%.1f' % (float(line[1]))).split('.')
        type = lookup[type]
        out = [line[0], type, float(temp)] + map(float, line[2:])
        return out

    def convert_dil(self, line):
        DIL = map(float, (line[3], line[4], line[2]))
        XYZ = DIL2XYZ(DIL)
        return line[:2] + list(XYZ)


def paterson_test():
    tdt_file =  os.path.join(RockPy.PaleoMag.test_data_path, '187A.tdt')
    # print tdt(sample_name='ET2_187A', dfile=tdt_file).formatted_data()
    sample = RockPy.Sample(name='ET2_187A')
    measurement = sample.add_measurement(ftype='tdt', mtype='thellier', fpath=tdt_file)
    return sample, measurement


if __name__ == '__main__':
    s, m = paterson_test()

