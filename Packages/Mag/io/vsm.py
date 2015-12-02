__author__ = 'mike'
from time import clock
import RockPy3
import numpy as np

from RockPy3.core import io
from os.path import join
from copy import deepcopy

class Vsm(io.ftype):
    def __init__(self, dfile, dialect=None):
        super(Vsm, self).__init__(dfile=dfile, dialect=dialect)
        self.raw_data = self.simple_import()
        # successively remove data from the raw_data
        self.micromag = self.raw_data.pop(0)
        self.mtype = self.raw_data.pop(0)

        # remove last line with ' MicroMag 2900/3900 Data File ends'
        self.raw_data.pop(-1)

        self.segment_start_idx = [i for i, v in enumerate(self.raw_data) if v.startswith('Segment')][0]

        # get the info header raw data
        # the data starts from line 1 up to the line where it sais 'Segments'
        self.info_header_raw = [self.raw_data.pop(0) for i in range(0, self.segment_start_idx)][1:]
        self.info_header = self.get_measurement_infos()

        self.calibration_factor = self.info_header['calibration factor']

        # remove all data points from raw data
        self.data_idx = min(i for i, v in enumerate(self.raw_data) if v.startswith('+') or v.startswith('-'))
        self._data = [self.raw_data.pop(self.data_idx) for i in range(self.data_idx, len(self.raw_data) - 1)]

        # indices have changed
        self.segment_start_idx = [i for i, v in enumerate(self.raw_data) if v.startswith('Segment')][0]
        self.segment_end_idx = min(i for i, v in enumerate(self.raw_data) if not v )
        self.segment_raw = [self.raw_data.pop(self.segment_start_idx) for i in range(self.segment_start_idx, self.segment_end_idx)]

        self.header, self.units = self.get_header()
        # micromag header with all the settings
        self.measurement_info = self.get_measurement_infos()

    @property
    def temperature(self):
        if self.measurement_info['temperature (measured)'] != 'N/A':
            return self.measurement_info['temperature (measured)']

    def get_segments_from_data(self):
        # the length of each field is calculated using the last line of the segments.
        # That line has a ',' at the point where we want to break, the index of the comma is used
        # to separate the line
        field_lengths = [0] + [i for i, v in enumerate(self.segment_raw[-1]) if v == ',']
        field_lengths.append(len(self.segment_raw[0]))

        # separate and join the lines
        seg_text = [i for i in self.segment_raw if not i[0].isdigit() if i]

        # split lines
        seg_text = [[seg[v:field_lengths[i+1]].strip()
                    for seg in seg_text] for i, v in enumerate(field_lengths[:-1])]
        # join texts
        seg_text = [' '.join(i).lower().rstrip() for i in seg_text]

        seg_nums = [i for i in self.segment_raw if i[0].isdigit()]
        # seg_text = list(map(str.lower, seg_text))
        # # convert and
        seg_nums = [map(self.convert2float_or_str, j.split(',')) for j in seg_nums]
        seg_nums = list(map(list, zip(*seg_nums)))

        # quick check if this makes sense
        if len(seg_nums[0]) != self.measurement_info['number of segments']:
            self.log.error('NUMBER OF SEGMENTS IS WRONG')

        return dict(zip(seg_text, seg_nums))

    def get_data(self):
        # get the empty line numbers
        empty_lines = [0]+[i for i,v in enumerate(self._data) if not v]+[len(self._data)]
        data = [np.array([i.split(',') for i in self._data[v: empty_lines[i+1]] if i]).astype(float)
                for i,v in enumerate(empty_lines[:-1])]

        return data

    def get_header(self):
        # the length of each field is calculated using the last line of the segments.
        # That line has a ',' at the point where we want to break, the index of the comma is used
        # to separate the line
        header_text = [i for i in self.raw_data if i]
        field_lengths = [0] + \
                        [i for i, v in enumerate(self._data[0]) if v == ','] + \
                        [len(self._data[0])]
        # # get text from headers
        header_text = [i.replace('\udcb2', ' 2').replace('Am', 'A m').replace('(', '').replace(')', '')  # remove brackets and change uft
                       for i in header_text]

        header_text = [[j[v: field_lengths[i + 1]].strip() for j in header_text] + [i] for i, v in
                       enumerate(field_lengths[:-1])]

        for i in range(len(header_text)):
            header_text[i] = [' '.join(header_text[i][:-2]).strip(), header_text[i][-2], header_text[i][-1]]

        return [i[0].lower() for i in header_text], [i[-2] for i in header_text]

    @staticmethod
    def split_comma_float(item):
        if item:
            return map(float, item.split(','))

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
                    out = (line[:splitter].rstrip().lower(), self.convert2float_or_str(line[splitter + 1:].strip()))

                    if out[1] == 'Yes':
                        return (line[:splitter].rstrip().lower(), True)
                    if out[1] == 'No':
                        return (line[:splitter].rstrip().lower(), False)
                    else:
                        return out


        data = self.info_header_raw
        data = [i for i in map(separate, data) if i and i[1]]
        return dict(data)

    def check_calibration_factor(self):
        pass