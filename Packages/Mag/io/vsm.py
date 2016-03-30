__author__ = 'mike'
from time import clock
import RockPy3
import numpy as np

from RockPy3.core import io


class Vsm(io.ftype):
    standard_calibration_exponent = 0

    @staticmethod
    def get_segment_start_idx(data):
        for i, v in enumerate(data):
            if v.startswith('Segment'):
                return i

    @staticmethod
    def get_data_idx(data):
        for i, v in enumerate(data):
            if v.startswith('+') or v.startswith('-'):
                return i

    @staticmethod
    def split_data(data):
        out = []
        aux = []
        for l in data:
            if not l:
                out.append(aux)
                aux = []
            else:
                l = [float(i) for i in l.split(',')]
                aux.append(l)
        return np.array(out)

    @staticmethod
    def get_segment_raw(data):
        out = []
        for i in data:
            if not i:
                return out
            else:
                out.append(i)

    def __init__(self, dfile, dialect=None):
        super(Vsm, self).__init__(dfile=dfile, dialect=dialect)
        raw_data = self.simple_import()

        # successively remove data from the raw_data
        self.micromag = raw_data.pop(0)
        self.mtype = raw_data.pop(0)

        # remove last line with ' MicroMag 2900/3900 Data File ends'
        raw_data.pop(-1)

        # get the index of the first segment line. Does not exist for FORC measurements
        if self.mtype == 'Direct moment vs. field; First-order reversal curves':
            self.segment_start_idx = self.get_data_idx(raw_data)-2
        else:
            self.segment_start_idx = self.get_segment_start_idx(raw_data)

        # get the info header raw data
        # the data starts from line 1 up to the line where it says 'Segments'
        raw_info_header = raw_data[1:self.segment_start_idx + 1]

        raw_data = raw_data[self.segment_start_idx:]  # remove header from raw_data

        # micromag header with all the settings
        self.info_header = self.get_measurement_infos(raw_info_header=raw_info_header)

        # check the calibration factor
        self.calibration_factor = self.info_header['calibration factor']

        if np.floor(np.log10(self.calibration_factor)) != self.standard_calibration_exponent:
            self.correct_exp = np.power(10, np.floor(np.log10(self.calibration_factor)))
            RockPy3.logger.warning(
                'CALIBRATION FACTOR (cf) seems to be wrong. Generally the exponent of the cf is {} here: {}. Data is corrected'.format(
                    self.standard_calibration_exponent,
                    int(np.floor(np.log10(self.calibration_factor)))))
        else:
            self.correct_exp = None

        # remove all data points from raw data
        self.data_idx = self.get_data_idx(raw_data)
        segment_raw = self.get_segment_raw(raw_data[:self.data_idx])

        if self.mtype == 'Direct moment vs. field; First-order reversal curves':
            # for FORC files there is no segment info
            self.header, self.units = self.get_header(raw_data[:self.data_idx])
        else:
            self.header, self.units = self.get_header(raw_data[len(segment_raw):self.data_idx])

        self.data = self.get_data(raw_data[self.data_idx:-1])
        self.segment_data = self.get_segments_from_data(segment_raw=segment_raw)

    def get_segments_from_data(self, segment_raw):
        # the length of each field is calculated using the last line of the segments.
        # That line has a ',' at the point where we want to break, the index of the comma is used
        # to separate the line
        field_lengths = [0] + [i for i, v in enumerate(segment_raw[-1]) if v == ',']
        field_lengths.append(len(segment_raw[0]))

        # separate and join the lines
        seg_text = [i for i in segment_raw if not i[0].isdigit() if i]

        # split lines
        seg_text = [[seg[v:field_lengths[i + 1]].strip()
                     for seg in seg_text] for i, v in enumerate(field_lengths[:-1])]
        # join texts
        seg_text = [' '.join(i).lower().rstrip() for i in seg_text]

        seg_nums = [i for i in segment_raw if i[0].isdigit()]
        # seg_text = list(map(str.lower, seg_text))
        # # convert and
        seg_nums = [map(self.convert2float_or_str, j.split(',')) for j in seg_nums]
        seg_nums = list(map(list, zip(*seg_nums)))

        # quick check if this makes sense
        if len(seg_nums[0]) != self.info_header['number of segments']:
            self.log.error('NUMBER OF SEGMENTS IS WRONG')

        return dict(zip(seg_text, seg_nums))

    def get_data(self, raw_data):
        # get the empty line numbers
        empty_lines = [0] + [i for i, v in enumerate(raw_data) if not v] + [len(raw_data)]
        data = np.array([np.array([i.split(',') for i in raw_data[v: empty_lines[i + 1]] if i]).astype(float)
                         for i, v in enumerate(empty_lines[:-1])])
        # data = self.split_data(self._data)
        if self.correct_exp:
            moment_idx = [i for i, v in enumerate(self.header) if v in ('moment', 'remanence', 'induced')]
            for idx in moment_idx:
                for i, d in enumerate(data):
                    data[i][:, idx] *= self.correct_exp

        return data


    @staticmethod
    def get_header(data):
        # columnwidth is assumed to be 14
        column_no = len(max(data))//14
        # calculate the splits for the maximal legnth in the header
        splits = tuple((i*14, (i+1)*14) for i in range(column_no+1))

        data = [[d[i[0]: i[1]].strip() for i in splits] for d in data if d]

        units = [i.replace('(','').replace(')','') for i in data[-1]]

        # correct Am^2 sign
        for i, v in enumerate(units):
            if 'Am' in v:
                units[i] = 'A m^2'

        header = [' '.join([d[n] for d in data[:-1] if d[n]]).lower() for n,v in enumerate(data[0])]
        header = [i for i in header if i]
        units = [i for i in units if i]
        return header, units

    @staticmethod
    def split_comma_float(item):
        if item:
            return map(float, item.split(','))

    def get_measurement_infos(self, raw_info_header):
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

        t = [i.split('  ') for i in raw_info_header]
        t = [[j for j in i if j] for i in t]
        t = [tuple(i) for i in t if len(i) > 1]
        data = {i[0].lower(): i[1] for i in t}

        for k, v in data.items():
            with RockPy3.ignored(ValueError):
                data[k] = float(v)
            if v == 'Yes':
                data[k] = True
            if v == 'No':
                data[k] = False

        return dict(data)

    def check_calibration_factor(self):
        pass


if __name__ == '__main__':
    from pprint import pprint
    forc_file = '/Users/mike/Downloads/FeNi20Mike/FeNi20H/FeNi_FeNi20-Ha36e02-G01_FORC_VSM#[]_[]_[]#-.002'
    hys_file = '/Users/mike/Dropbox/experimental_data/FeNiX/FeNi20KSand/Backups/FeNi_FeNi20-Ka006-G02_HYS_VSM#1,35[mg]_[]_[]##STD032.001'

    # t = Vsm(dfile=hys_file)
    t = Vsm(dfile=forc_file)
