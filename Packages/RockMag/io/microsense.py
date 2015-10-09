__author__ = 'mike'
import numpy as np

from RockPy.core import io


class MicroSense(io.ftype):
    def __init__(self, dfile, sample_name):
        super(MicroSense, self).__init__(dfile, sample_name)

        self.raw_data = [i for i in open(self.file_name)]

        self.data_start_idx = [i + 2 for i,v in enumerate(self.raw_data) if self.raw_data[i].startswith('@@Final')][0]
        self.data_end_idx = [i - 2 for i,v in enumerate(self.raw_data) if self.raw_data[i].startswith('@@END Data')][-1]
        self.header_idx = [i + 1 for i,v in enumerate(self.raw_data) if self.raw_data[i].startswith('@@End of Header')][
            0]
        d = np.array([i.split() for i in self.raw_data[self.data_start_idx:self.data_end_idx]])
        # print self.data_start_idx, self.data_end_idx
        self.data = d.astype(float)
        # self.data = rockpydata(column_names=self.header, data=d)

    @property
    def header(self):
        return self.raw_data[self.header_idx].lower().split()

    def out_hys(self):
        return self.data
