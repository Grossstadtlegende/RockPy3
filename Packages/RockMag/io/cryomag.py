__author__ = 'volk'
import datetime

import numpy as np
import matplotlib.dates

from RockPy.core import io


class CryoMag(io.ftype):
    def __init__(self, dfile, sample_name):
        super(CryoMag, self).__init__(dfile, sample_name)
        # self.raw_data = self.simple_import()
        self.raw_data = map(self.split_tab , self.simple_import())

        # self.raw_data = [i.strip('\n\r').split('\t') for i in open(self.file_name).readlines()]

        d = np.array(self.raw_data[2:])
        self.float_data_idx = [7, 12, 13, 14, 16]#, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20
        self.time_idx = 10

        if sample_name == 'test_sample':
            sample_name = self.samples[0]
        try:
            self.sample_idx = [i for i, v in enumerate(d[:, 0]) if
                               v == sample_name or sample_name in d[i, 9]]
        except IndexError:
            raise IndexError(sample_name)

        self.results_idx = [i for i, v in enumerate(d[:, 11]) if v == 'results' and i in self.sample_idx]
        self.data = np.array(
            [v for i, v in enumerate(d) if v[0] == sample_name or sample_name in v[9]
             if v[11] == 'results'
             if not '#' in v[0]])
        self.header = self.get_header()

    @property
    def float_data(self):
        return self.get_float_data()

    def get_header(self):
        header = [i.split('[Am^2]')[0] for i in self.raw_data[1]]
        header = map(str.strip, header)
        header = map(str.lower, header)
        header = np.array(header)
        return header

    @property
    def float_header(self):
        return self.header[self.float_data_idx]

    def get_float_data(self):
        # data = np.array(self.raw_data[2:])[self.results_idx]
        data = self.data[:, self.float_data_idx]
        data = data.astype(float)
        return np.array(data)

    def get_time_data(self):
        data = np.array(self.raw_data[2:])[self.results_idx]
        data = data[:, self.time_idx]
        data = map(self.convert_time, data)
        return np.array(data)

    @property
    def samples(self):
        samples = np.array(self.raw_data[2:])[:, 0]
        return samples

    def comments(self):
        comments = np.array(self.raw_data[2:])[:, 9]
        return comments

    @property
    def steps(self):
        steps = np.array(self.raw_data[2:])[:, 8]
        steps = np.array(map(str.lower, steps))
        steps = steps[self.results_idx]
        return steps

    def _check_data_exists(self):
        if len(self.sample_idx) != 0:
            return True
        else:
            return False #todo clever?

    def convert_time(self, time):
        '''
        converts a time string of the format : 14-02-13 17:12:25 into seconds
        :param time:
        :return:
        '''
        aux = datetime.datetime.strptime(time, "%y-%m-%d %H:%M:%S")
        out = matplotlib.dates.date2num(aux)
        return out