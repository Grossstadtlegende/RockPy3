__author__ = 'mike'
import numpy as np

from RockPy.core.data import RockPyData
from RockPy.core import io
from RockPy.utils.general import ignored

class Vsm(io.ftype):
    _calibration_exponent = 0
    # @profile()
    def __init__(self, dfile, sample=None):
        super(Vsm, self).__init__(dfile=dfile, sample_name=sample)

        with open(self.file_name, 'rU') as f:
            reader_object = f.readlines()

        self.measurement_header = self.readMicroMagHeader(reader_object)  # get header

        self.raw_out = [i for i in reader_object][self.measurement_header['meta']['numberoflines']:]  # without header

        self.header_idx = {v: i for i, v in enumerate(self.header)}

        self.data_header = [i for i in self.raw_out[0].split(' ') if i != '' if i != '\n']
        self.data_header_idx = {v.lower(): i for i, v in enumerate(self.data_header)}

        if self.measurement_header['MEASUREMENT']['Temperature (measured)']:
            self.temperature = self.measurement_header['MEASUREMENT']['Temperature (measured)']
        else:
            self.temperature = None

    @property
    def calibration_factor(self):
        # get calibration factor and calibration component for checking of exponent
        # sometimes someone might have msised a milli -> all data is factor 1000 wrong...
        return float(self.measurement_header['SETTINGS']['Calibration factor'])

    @property
    def calibration_exponent(self):
        return int(self.measurement_header['SETTINGS']['Calibration factor'].split('E')[1][1:])

    @property
    def header(self):
        """
        returns the actual data header, not th file/measurement header of the data
        """
        # vsm segemnts start with 'segment'
        new_line_idx = min(i for i, v in enumerate(self.raw_out) if v == '\n')  # all indices with new line

        # find first data index
        data_start_idx = min(i for i,v in enumerate(self.raw_out) if v.startswith('+') or v.startswith('-')) # first index of data

        data_header = self.raw_out[new_line_idx + 1: data_start_idx-1]
        data_header = [i for i in data_header]
        sifw = [len(i) + 1 for i in self.raw_out[data_start_idx].split(',')]
        sifw += [len(self.raw_out[data_start_idx])]
        sifw = [sum(sifw[:i]) for i in range(len(sifw))]
        data_header = np.array(
            [[v[sifw[i]: sifw[i + 1]] for i in range(len(sifw) - 1)] for j, v in enumerate(data_header)]).T

        data_header = [" ".join(i) for i in data_header]
        data_header = [' '.join(j.split()) for j in data_header]
        data_header = [j.split(' (')[0].lower() for j in data_header]
        return data_header

    @property
    def units(self):
        idx = min(i for i,v in enumerate(self.raw_out) if v.startswith('+') or v.startswith('-'))-1
        out = self.raw_out[idx].replace('\xb2', '^2').replace('\xb0', '').replace('(', '').replace(')', '').split()

        for i,v in enumerate(out):
            if v == 'Am^2': # Pint does not know Am there has to be a ' '
                out[i] = 'A m^2'

            if v == 'Am': # Pint does not know Am there has to be a ' '
                out[i] = 'A m'
        return out

    @property
    def segment_info(self):
        segment_start_idx = [i for i, v in enumerate(self.raw_out) if
                             v.strip().lower().startswith('segment') or v.strip().lower().startswith('number')]
        if segment_start_idx:
            segment_numbers_idx = [i for i, v in enumerate(self.raw_out) if v.startswith('0')]

            segment_data = [v.strip('\n').split(',') for i, v in enumerate(self.raw_out) if i in segment_numbers_idx]
            sifw = [len(i) + 1 for i in segment_data[0]]
            sifw += [len(segment_data[0])]
            sifw = [sum(sifw[:i]) for i in range(len(sifw))]
            segment_data = np.array(segment_data).astype(float)
            segment_info = np.array(
                [[v[sifw[i]: sifw[i + 1]] for i in range(len(sifw) - 1)] for j, v in enumerate(self.raw_out) if
                 j in segment_start_idx]).T
            segment_info = [' '.join(i) for i in segment_info]
            segment_info = np.array([' '.join(j.split()).lower() for j in segment_info])
            out = RockPyData(column_names=segment_info, data=segment_data)
        else:
            out = None
        return out

    @property
    def out(self):
        # ### header part
        data_header = [i.split('\n')[0] for i in self.raw_out if
                       not i.startswith('+') and not i.startswith('-') and not i.split() == []][:-1]
        # getting first data line
        data_start_idx = [i for i, v in enumerate(self.raw_out) if
                          v.strip().lower().startswith('+') or v.strip().lower().startswith(
                              '-')][0]  # first idx of all indices with + or - (data)

        # setting up all data indices
        data_indices = [data_start_idx] + [data_start_idx + i for i in
                                           list(map(int, self.segment_info['final index'].v))] + [
                           len(self.raw_out[data_start_idx:]) + data_start_idx-1]

        data = [self.raw_out[data_indices[i]:data_indices[i + 1]] for i in range(len(data_indices) - 1)]

        data = [[j.strip('\n').split(',') for j in i if not j == '\n'] for i in data]
        data = [np.array([map(float, j) for j in i]) for i in data]

        if 'temperature' in self.header:
            self.temperature = np.mean([j[self.header_idx['temperature']] for i in data for j in i])
        else:
            self.temparature = None

        # reformating to T / Am2 / Celsius
        if self.measurement_header['INSTRUMENT']['Units of measure'] == 'cgs':
            for i in range(len(data)):
                data[i][:, 1] *= 1e-3  # emu to Am2
                data[i][:, self.header_idx['field']] *= 1e-4  # oe to T

        if self.measurement_header['INSTRUMENT']['Temperature in'] == 'Kelvin':
            for i in range(len(data)):
                # data[i][:,] *= 1e-3 # emu to Am2
                with ignored(KeyError):
                    data[i][:, self.header_idx['temperature']] -= 0  # 273.15 # K to C
            # if not 'temperature' in
            # self.log.debug('No temperature data stored')

        if Vsm._calibration_exponent != self.calibration_exponent:
            data = self.correct_wrong_exponent(data=data)
        return data

    def correct_wrong_exponent(self, data):
        self.log.warning('CALIBRATION EXPONENT does not match << %s != 0 >> check data, using << E+00 >>' %self.calibration_exponent)
        for k,v in self.header_idx.iteritems():
            if k not in ['field', 'temperature', 'time']:
                for i in range(len(data)):
                    data[i][:, v] *= 10**(Vsm._calibration_exponent - self.calibration_exponent)  # correct wrong conversion factor
        return data

    def forc(self):
        out = [i for i in self.raw_out if i.startswith('+') or i.startswith('-') or i.split() == []]
        out_data = []
        aux = []
        for i in out:
            if len(i) != 1:
                if i.strip() != '':
                    d = i.strip('\n').split(',')
                    try:
                        d = map(float, d)
                        aux.append(d)
                    except:
                        if 'Adjusted' in d[0].split():
                            adj = True
                        pass
            else:
                aux = np.array(aux)
                out_data.append(aux)
                aux = []
        out_data = np.array(out_data)
        return out_data

    def readMicroMagHeader(self, lines):
        sectionstart = False
        sectiontitle = None
        sectioncount = 0
        header = {}
        # section header: CAPITALS, no spaces
        lc = 0
        for l in lines:
            lc += 1
            sl = l.strip()  # take away any leading and trailing whitespaces
            if lc == 1 and not sl.startswith("MicroMag 2900/3900 Data File"):  # check first line
                # self.log.error("No valid MicroMag file. Header not found in first line.")
                return None

            if len(sl) == 0:  # empty line
                sectionstart = True
                continue  # go to next line
            if sectionstart:  # previous line was empty
                sectionstart = False
                if sl.isupper():  # we have a capitalized section header
                    # log.INFO('reading header section %s' % sl)
                    sectiontitle = sl
                    header[sectiontitle] = {}  # make new dictionary to gather data fields
                    sectioncount += 1
                    continue  # go to next line
                else:  # no captitalized section header after empty line -> we are probably at the end of the header
                    # verbous.INFO('reading header finished at line %d' % lc)
                    break  # end of header
            if sectiontitle != None:  # we are within a section
                # split key value at fixed character position
                key = sl[:31].strip()
                value = sl[31:].strip(' "')
                if len(key) != 0:
                    header[sectiontitle][key] = value
        header['meta'] = {}
        header['meta']['numberoflines'] = lc - 1  # store header length
        header['meta']['numberofsections'] = sectioncount
        return header

    def out_thermocurve(self):
        return self.out

    def out_backfield(self):
        return self.out

    def out_hysteresis(self):
        return self.out

    def _check_data_exists(self):
        if self.raw_out:
            return True
        else:
            return False