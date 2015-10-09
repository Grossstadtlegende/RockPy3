__author__ = 'mike'
from time import clock
import RockPy
import numpy as np

from RockPy.core import io
from os.path import join
import vsm
from profilehooks import profile
from RockPy.utils.general import ignored


class VsmNew(io.ftype):
    # @profile()
    def __init__(self, dfile, sample=None):
        super(VsmNew, self).__init__(dfile=dfile, sample_name=sample)

        with open(dfile) as f:
            self.raw_data = map(str.rstrip, f.readlines())

        # self.data_idx = min(i for i, v in enumerate(self.raw_data) if v.startswith('+') or v.startswith('-'))
        self.data_idx = self.get_data_idx()

        self.fformat_header = self.get_file_header()
        self.segment_header = self.get_segment_header()

        self.data = self.get_data()

    def get_data_idx(self):
        for i, v in enumerate(self.raw_data):
            if v.startswith('+') or v.startswith('-'):
                return i

    def get_file_header(self):
        out = {}
        linecount = 0
        header_lines = [i.split('  ') for i in self.raw_data[3:self.data_idx]]
        header_lines = (filter(None, i) for i in header_lines)

        key = 'ftype'
        for idx, line in enumerate(header_lines):
            if len(line) == 1:
                if len(line[0].split(',')) == 1:
                    key = line[0]
            out.setdefault(key, dict())
            if len(line) == 2 and not all(i.strip() == 'Adjusted' for i in line):
                out[key].setdefault(line[0], line[1])
                linecount = idx + 4
        self.segment_index = linecount
        return out

    def get_segment_header(self):
        if 'NForc' in self.fformat_header['SCRIPT']:
            return

        # get number of segments
        segments = int(self.fformat_header['SCRIPT']['Number of segments'])
        # get end index, the last '\r\n' entry
        segment_end_idx = max(i for i, v in enumerate(self.raw_data[self.segment_index:self.data_idx]) if not v)
        segment_info = [i.rstrip() for i in
                        self.raw_data[self.segment_index + 1:self.segment_index + segment_end_idx - segments]]
        # get length of each entry from data lengths
        segment_length = [len(i) + 1 for i in self.raw_data[self.segment_index + segment_end_idx - segments].split(',')]

        # cut entries to represent length
        for i, v in enumerate(segment_info):
            idx = 0
            aux = []
            for j, length in enumerate(segment_length):
                aux.append(v[idx:idx + segment_length[j]].strip())
                idx += segment_length[j]
            segment_info[i] = aux

        # filter empty lists and create generator
        # join segment names ( segment\nnumber -> segment_number
        segment_info = [' '.join(filter(len, i)) for i in zip(*segment_info)]

        segments = [map(float, i.rstrip().split(',')) for i in
                    self.raw_data[self.segment_index + segment_end_idx - segments:self.segment_index + segment_end_idx]]

        # initialize dictionary
        out = {v: [j[i] for j in segments] for i, v in enumerate(segment_info)}
        return out

    def get_data(self):
        data = self.raw_data[self.data_idx:-2]
        # different readin procedure for Forc Data, because no segment information
        if 'NForc' in self.fformat_header['SCRIPT']:
            indices = [-1] + [i for i, v in enumerate(data) if not v] + [len(data)]
            # out = [map(self.split_comma_float, [line for line in data[indices[i]+1:indices[i+1]]]) for i in xrange(len(indices)-1)]
            out = [np.asarray([line.split(',') for line in data[indices[i] + 1:indices[i + 1]]], dtype=float) for i in
                   xrange(len(indices) - 1)]
        else:
            indices = [0] + map(int, self.segment_header['Final Index'])
            data = filter(len, data)  # remove empty lines
            out = [np.asarray([line.split(',') for line in data[indices[i]:indices[i + 1]]], dtype=float) for i in
                   xrange(len(indices) - 1)]
        return out


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


if __name__ == '__main__':
    dfile = join(RockPy.test_data_path, 'vsm', 'LTPY_527,1a_HYS_VSM#[]_[]_[]#TEMP_300_K#STD000#.000')
    dfile = '/Users/mike/Dropbox/experimental_data/FORC/LF4/LF4_Ve_HYS_FORC-__-.001'
    start = clock()
    old = vsm.Vsm(dfile=dfile)
    old.forc()
    print 'old readin', clock() - start

    start = clock()
    new = VsmV2(dfile=dfile)
    # print new.data[0]
    # print new.segement_info
    # print new.micromag
    # print new.get_measurement_infos()
    print 'new', clock() - start
