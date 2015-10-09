__author__ = 'mike'

from string import rstrip

import numpy as np

from RockPy.core import io


# todo get rid of most functions

class SushiBar(io.ftype):
    float_data_idx = [5, 6, 7, 18, 19, 24, 31, 8, 30, 15, 28, 27, 26, 25, 16, 29, 11, 14, 9, 17,
                      22, 10, 13, 12]
    header = {'sample': 0, 'site': 1, 'type': 2, 'run': 3, 'time': 4, 'x': 5, 'y': 6, 'z': 7,
              'm': 8, 'dc': 9, 'ic': 10, 'dg': 11, 'ig': 12, 'ds': 13, 'is': 14, 'a95': 15,
              'sm': 16, 'npos': 17, 'dspin': 18, 'ispin': 19, 'holder/sample': 20, 'cup/sample': 21,
              'bl diff/sample': 22, 'steps/rev': 23, 'par1': 24, 'par2': 25, 'par3': 26, 'par4': 27,
              'par5': 28, 'par6': 29, 'strat_level': 30, 'geoaz': 31, 'hade': 32, 'dipdir': 33,
              'dip': 34, 'HeaderSkip': 1, 'modes': [], 'mode': 1, 'step': 25, }

    float_header = ['x', 'y', 'z', 'dspin', 'ispin', 'par1', 'dip', 'dipdir', 'geoaz', 'm', 'strat_level',
                    'a95', 'par5', 'par4', 'par3', 'par2', 'sm', 'par6', 'dg', 'is', 'hade', 'dc', 'npos',
                    'bl diff/sample', 'ic', 'ds', 'ig']

    def __init__(self, dfile, sample_name):
        super(SushiBar, self).__init__(dfile, sample_name)

        with open(self.file_name) as f:
            data = map(rstrip, f)
            data = [i.split('\t') for i in data]

        self.sample_names = list(set([i[0] for i in data[1:]]))
        self.sample_names = map(str, self.sample_names)

        self.raw_data = [i for i in data if i[0] in [sample_name, 'test_sample']
                         and i[8] != 'None'
                         and i[8] != '-']  # filter for sample and missing data
        if len(self.raw_data) == 0:
            self.log.error('SAMPLE NAME << %s >> not recognized: SAMPLES: %s' % (sample_name, str(self.sample_names)))
            # base.Machine.logger.error('SAMPLE NAME not recognized: SAMPLES: %s' % str(self.sample_names))
            raise KeyError
        self.raw_data = np.array(map(self.__replace_none, self.raw_data))

    def __replace_none(self, line):
        """
        takes a line and replaces none and '-' strings with numpy.nan
        :param line:
        :return: line
        """
        for i, v in enumerate(line):
            if v == 'None':
                line[i] = '0'
            if v == '-':
                line[i] = np.nan
        return line

    def get_data(self, key):
        idx = SushiBar.header[key]
        return self.raw_data[:, idx]

    def float_data(self):
        return self.raw_data[:, SushiBar.float_data_idx].astype(float)

    def get_header(self):
        out = ['sample', 'site', 'type', 'run', 'meas. time', 'x', 'y', 'z', 'm', 'dc', 'ic', 'dg', 'ig', 'ds', 'is',
               'a95', 'sm', 'npos', 'dspin', 'ispin', ' holder/sample', 'cup/sample', 'bl diff/sample', 'steps/rev',
               'par1', 'par2', 'par3', 'par4', 'par5', 'par6', 'strat_level', 'geoaz', 'hade', 'dipdir', 'dip']
        return out

    def out_afdemag(self):
        idx = [24, 5, 6, 7]
        return self.raw_data[:, idx].astype(float)

    def out_parm_spectra(self):
        idx = [24, 25, 26, 27, 28, 5, 6, 7]
        header = ['ac_field', 'dc_field', 'upper_window', 'lower_window', 'step', 'x', 'y', 'z']
        units = ['mT', 'mT', 'mT', 'mT', 'mT', 'A m^2', 'A m^2', 'A m^2']
        return self.raw_data[:, idx].astype(float), header, units

    def out_trm(self):
        idx = [24, 5, 6, 7, 16]
        return self.raw_data[:, idx].astype(float)

    def _check_data_exists(self):
        if len(self.raw_data) != 0:
            return True
        else:
            return False


def test():
    import RockPy
    from os.path import join

    test_file = join(RockPy.test_data_path, 'MUCSUSH_af_test.af')
    S = RockPy.Sample(name='WURM')
    print S.add_measurement(mtype='afdemag', fpath=test_file, ftype='sushibar')


if __name__ == '__main__':
    test()
