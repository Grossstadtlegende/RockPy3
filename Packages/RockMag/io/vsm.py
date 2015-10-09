__author__ = 'mike'
from time import clock
import RockPy
import numpy as np

from RockPy.core import io
from os.path import join
import vsm
from profilehooks import profile
from RockPy.utils.general import ignored

class VsmV2(io.ftype):
    def __init__(self, dfile, dialect=None):
        super(VsmV2, self).__init__(dfile=dfile, dialect=dialect)
        self.raw_data = self.simple_import()
        self.micromag = self.raw_data.pop(0)
        self.mtype = self.raw_data.pop(0)

        # remove last line with ' MicroMag 2900/3900 Data File ends'
        self.raw_data.pop(-1)
        # micromag header with all the settings
        self.segment_start_idx, self.data_idx = self.segments_data_idx()
        self.measurement_info = self.get_measurement_infos()
        self.segment_end_idx = self.segment_start_idx + int(self.measurement_info['number of segments']) + 3
        self.segement_info = self.get_segments_from_data()

        self.data = self.get_data()
        self.header = self.get_header()

    def get_segments_from_data(self):
        # the length of each field is calculated using the last line of the segments.
        # That line has a ',' at the point where we want to break, the index of the comma is used
        # to separate the line
        # todo is it always three lines in the segment part?
        field_lengths = [0] + [i for i, v in enumerate(self.raw_data[self.segment_end_idx - 1]) if v == ',']
        field_lengths.append(len(self.raw_data[self.segment_end_idx - 1]))

        # separate and join the lines
        seg_text = self.raw_data[self.segment_start_idx:self.segment_start_idx + 3]
        seg_text = [' '.join([j[v: field_lengths[i + 1]].strip() for j in seg_text if j]).rstrip() for i, v in
                    enumerate(field_lengths[:-1])]
        seg_text = map(str.lower, seg_text)
        # convert and
        seg_nums = [map(self.convert2float_or_str, j.split(',')) for j in
                    self.raw_data[self.segment_start_idx + 3:self.segment_end_idx]]
        seg_nums = map(list, zip(*seg_nums))

        # quick check if this makes sense
        if len(seg_nums[0]) != self.measurement_info['number of segments']:
            self.log.error('NUMBER OF SEGMENTS IS WRONG')
        return dict(zip(seg_text, seg_nums))

    def get_data(self):

        # todo is it always two lines in the segment part?
        s_index = [self.data_idx + 2] + [int(v)+i + self.data_idx + 2 for i,v in
                                         enumerate(self.segement_info['final index'])]
        e_index = [v + i for i, v in enumerate(s_index[1:])]
        index = zip(s_index, e_index)
        data = [[map(self.convert2float_or_str, i.split(',')) for i in self.raw_data[i[0]:i[1]] if i] for i in index]
        # transpose the data
        data =  [map(list, zip(*i)) for i in data]
        return data

    def get_header(self):
        # the length of each field is calculated using the last line of the segments.
        # That line has a ',' at the point where we want to break, the index of the comma is used
        # to separate the line
        field_lengths = [0] + \
                        [i for i, v in enumerate(self.raw_data[self.data_idx + 2]) if v == ',']+\
                        [len(self.raw_data[self.data_idx + 2])]

        # get text from headers
        header_text = [i.replace('\xb2', ' 2').replace('(','').replace(')','') # remove brackets and change uft
                       for i in self.raw_data[self.data_idx:self.data_idx + 2]]
        header_text = [[j[v: field_lengths[i + 1]].strip() for j in header_text]+[i] for i, v in
                       enumerate(field_lengths[:-1])]

        return {i[0]:i[1:] for i in header_text}

    @staticmethod
    def split_comma_float(item):
        if item:
            return map(float, item.split(','))

    def segments_data_idx(self):
        out = []
        for i, v in enumerate(self.raw_data):
            if 'Segment' in v:
                out.append(i)
            if 'Field' in v and 'Moment' in v:
                out.append(i)
                return out

    def get_measurement_infos(self):
        """
        takes the raw data and creates a dictionary with the measurement infos
        """

        def separate(line):
            if line:
                if any(j in line for j in ('+', '-', '\"')):
                    splitter = 30
                else:
                    splitter = 31
                if line[splitter + 1:].rstrip():
                    return (line[:splitter].rstrip().lower(), self.convert2float_or_str(line[splitter + 1:].rstrip()))

        data = self.raw_data[:self.segment_start_idx]
        data = [i for i in map(separate, data) if i and i[1]]
        return dict(data)
