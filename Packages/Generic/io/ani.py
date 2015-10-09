__author__ = 'wack'

from StringIO import StringIO

import numpy as np

from RockPy.core import io


class Ani(io.ftype):
    """
    ani file format to describe anisotropy measurements
    format:
    first line header: sample name; type (IRM / ARM / TRM / SUS); window (90-0); magnetizing field; comment
    e.g.: ABC; ARM; 50-40; 0.1; no comment
    second line: reference/measurement directions in pairs of declination and inclinations (mdirs)
    i.e. D1,I1;D2,I2;D3,I3
    remaining lines 1 or 3 components of measurements
    i.e. V or x,y,z
    """

    def __init__(self, dfile, sample_name):
        super(Ani, self).__init__(dfile=dfile, sample_name=sample_name)

        # read the data file line by line
        f = open(self.file_name)
        self.raw_data = f.readlines()
        f.close()

        # store first header line
        self.header = self.raw_data[0]
        dipairs = self.raw_data[1].strip().split(';')
        self.mdirs = [[float(di[0]), float(di[1])] for di in [dipair.split(',') for dipair in dipairs]]
        # read the directional measurements from remaining lines
        self.data = np.genfromtxt(StringIO(''.join(self.raw_data[2:])), delimiter=",")



