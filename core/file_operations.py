from RockPy3.utils.general import add_unit

__author__ = 'mike'
import os
import os.path
from os.path import expanduser, join
from collections import defaultdict
from copy import deepcopy
import numpy as np
from pint import UnitRegistry
import RockPy3
import pickle
import RockPy3.core

ureg = UnitRegistry()
default_folder = join(expanduser("~"), 'Desktop', 'RockPy')

import os, re

def mtype_ftype_abbreviations():
    RockPy3.logger.debug('READING FTYPE/MTYPE abbreviations')
    with open(join(RockPy3.installation_directory, 'abbreviations')) as f:
        abbrev = f.readlines()
    abbrev = [tuple(i.rstrip().split(':')) for i in abbrev if i.rstrip() if not i.startswith('#')]
    abbrev = dict((i[0], [j.lstrip() for j in i[1].split(',')]) for i in abbrev)
    inv_abbrev = {i: k for k in abbrev for i in abbrev[k]}
    inv_abbrev.update({k: k for k in abbrev})
    return inv_abbrev, abbrev

def abbreviate_name(name):
    if name:
        return RockPy3.mtype_ftype_abbreviations[name.lower()][0]


def fname_abbrevs():
    abbrev = RockPy3.mtype_ftype_abbreviations
    out = {}
    for k, v in abbrev.items():
        out.setdefault(k, v[0].upper())
        if len(v) > 1:
            for i in v[1:]:
                out.setdefault(i, v[0].upper())
    return out


def save(smthg, file_name, folder=None):
    if not folder:
        folder = default_folder
    with open(join(folder, file_name), 'wb+') as f:
        # dump = numpyson.dumps(smthg)
        dump = pickle.dumps(smthg)
        f.write(dump)
        f.close()


def abbrev_to_classname(abbrev):
    """
    takes an abbreviated classname e.g. 'hys' and returns the true class_name
    """
    abbrev = abbrev.rstrip().lower()
    if not abbrev:
        return ''
    try:
        class_name = RockPy3.mtype_ftype_abbreviations_inversed[abbrev.lower()]
    except KeyError:
        raise KeyError('{} not in abbrev -> class_name dictionary'.format(abbrev))
    return class_name


def name_to_abbrev(name):
    if not name:
        return ''
    return RockPy3.mtype_ftype_abbreviations[name.lower()][0]


def convert_to_fname_abbrev(name_or_abbrev):
    if name_or_abbrev:
        name = abbrev_to_classname(name_or_abbrev)
        abbrev = name_to_abbrev(name)
        return abbrev.upper()
    else:
        return ''


def load(file_name, folder=None):
    # print 'loading: %s' %join(folder, file_name)
    if not folder:
        folder = default_folder
    with open(join(folder, file_name), 'rb') as f:
        # out = numpyson.loads(f.read())
        out = pickle.loads(f.read())
    return out


def save_xml(file_name=None, folder=None):
    """
    save master study to xml file
    :param file_name:
    :param folder:
    :return:
    """
    RockPy3.Study.save_xml(file_name, folder)


def load_xml(file_name, folder=None):
    """
    load master study from file
    :param file_name:
    :param folder:
    :return:
    """
    study = RockPy3.core.study.Study.load_from_xml(file_name, folder)
    return study


def get_fname_from_info(samplegroup='', sample_name='', #todo redundant
                        mtype='', ftype='',
                        mass='', mass_unit='',
                        height='', length_unit='',
                        diameter='',
                        series=None,
                        std=None,
                        idx=None,
                        **options):
    """
    generates a name according to the RockPy specific naming scheme.

    :param samplegroup: str
    :param sample_name: str
    :param mtype: str
    :param ftype: str
    :param mass: float
    :param mass_unit: str
    :param height: float
    :param length_unit: str
    :param diameter: float
    :param series: list, tuple
    :param std: str
    :param idx: int
    :param options:
    :return:
    """
    # abbrev = fname_abbrevs()
    mtype = abbrev_to_classname(mtype)
    ftype = abbrev_to_classname(ftype)

    options.pop('fpath', None)
    if not mtype.lower() in RockPy3.implemented_measurements:
        RockPy3.logger.warning('MEASUREMENT mtype << %s >> may be wrong or measurement not implemented, yet.' % mtype)

    if not ftype.lower() in RockPy3.Measurement.implemented_ftypes():
        RockPy3.logger.warning('MEASUREMENT ftype << %s >> may be wrong or ftype not implemented, yet.' % ftype)

    mtype = convert_to_fname_abbrev(mtype)
    ftype = convert_to_fname_abbrev(ftype)

    if idx is None:
        idx = '%03i' % (np.random.randint(999, size=1)[0])
    else:
        idx = str(idx)
        # make idx at least three digits long
        while len(idx) < 3:
            idx = '0' + idx

    if type(series) == tuple:
        aux = list()
        aux.append(series)
        series = aux

    if series:
        # check that stypes, svals, sunits are lists
        stypes = [s[0] for s in series]
        svals = [s[1] for s in series]
        svals = [str(i).replace('.', ',') for i in svals]  # replace '.' so no conflict with file ending
        sunits = [s[2] for s in series]
    else:
        stypes, svals, sunits = [], [], []

    sample = '_'.join([samplegroup, sample_name, mtype.upper(), ftype.upper()])

    if not any(i for i in [height, diameter]):
        length_unit = ''
    if not any(i for i in [mass]):
        mass_unit = ''
        mass = ''

    sample_info = '_'.join(
        [add_unit(str(mass).replace('.', ','), mass_unit),
         add_unit(str(diameter).replace('.', ','), length_unit),
         add_unit(str(height).replace('.', ','), length_unit),
         ])

    params = ';'.join(
        ['_'.join(map(str, [stypes[i], svals[i], sunits[i]])) for i in range(len(stypes))])

    if options:
        opt = ';'.join(['_'.join([k, str(v)]) for k, v in sorted(options.items())])
    else:
        opt = ''
    if std:
        std = 'STD%s' % str(std)
    else:
        std = ''

    out = '#'.join([sample, sample_info, params, std, opt])
    out += '.%s' % idx
    return out


def get_info_from_fname(path=None): # todo redundant
    """
    extracts the file information out of the filename

    Parameters
    ----------
       path:
          complete path, with folder/fname. Will be split into the two

    Raises
    ------
        KeyError if ftype or mtype not in RockPy3.mtype_ftype_abbreviations_inversed

    """
    # change add_measurement accordingly
    folder = os.path.split(path)[0]
    fname = os.path.split(path)[1]

    fpath = fname

    index = fname.split('.')[-1]
    fname = fname.split('.')[:-1][0]

    rest = fname.split('#')

    samplegroup, sample_name, mtype, ftype = rest[0].split('_')

    sample_info = [i.strip(']').split('[') for i in rest[1].split('_')]
    mass, diameter, height = sample_info

    if rest[2]:
        series = rest[2]
        # separate the series (combined with ;)
        series = series.split(';')
        # change , -> .

        series = [i.replace(',', '.').split('_') for i in series]
        series = [(i[0], float(i[1]), i[2]) for i in series]
    else:
        series = None
    try:
        STD = [int(i.lower().strip('std')) for i in rest if 'std' in i.lower()][0]
    except IndexError:
        STD = None

    try:
        options = [i.split('_') for i in rest[4].split('.')[0].split(';')]
        options = {i[0]: i[1] for i in options}
    except IndexError:
        options = None

    # convert mass to float
    with RockPy3.ignored(ValueError):
        mass[0] = mass[0].replace(',', '.')
        mass[0] = float(mass[0])

    # convert height to float
    with RockPy3.ignored(ValueError):
        diameter[0] = diameter[0].replace(',', '.')
        diameter[0] = float(diameter[0])

    # convert diameter to float
    with RockPy3.ignored(ValueError):
        height[0] = height[0].replace(',', '.')
        height[0] = float(height[0])

    if diameter[1] and height[1]:
        if diameter[1] != height[1]:
            diameter[0] = diameter[0] * getattr(ureg, height[1]).to(
                getattr(ureg, diameter[1])).magnitude

    mtype = mtype.lower()  # convert to upper for ease of checking
    ftype = ftype.lower()  # convert to upper for ease of checking

    try:
        mtype = RockPy3.mtype_ftype_abbreviations_inversed[mtype]
    except KeyError:
        raise KeyError('%s not implemented yet' % mtype)
        return

    try:
        ftype = RockPy3.mtype_ftype_abbreviations_inversed[ftype]
    except KeyError:
        raise KeyError('%s not implemented yet' % mtype)
        return

    # replace 'xml' ending with idx:
    if index == 'xml':
        index = 1

    out = {
        'samplegroup': samplegroup,
        'sample_name': sample_name,  # not needed since 3.5 rewrite
        'mtype': mtype,
        'ftype': ftype,
        'fpath': join(folder, fpath),
        'series': series,
        # 'std': STD,
        'idx': int(index),
        'mass': None,
        'mass_unit': None,
        'diameter': None,
        'height': None,
        'length_unit': None,
    }

    # if mtype == 'mass':
    if mass[0]:
        out.update({'mass': mass[0],
                    'mass_unit': mass[1]})
    # if mtype == 'diameter':
    if diameter[0]:
        out.update({'diameter': diameter[0],
                    'length_unit': diameter[1]})
    # if mtype == 'height':
    if height[0]:
        out.update({'height': height[0],
                    'length_unit': diameter[1]})
    if options:
        out.update(options)

    return out


def import_folder(folder, name='study', study=None):
    if not study:
        study = RockPy3.Study(name=name)

    files = [i for i in os.listdir(folder) if not i == '.DS_Store' if not i.startswith('#')]
    samples = defaultdict(list)

    for i in files:
        d = RockPy3.get_info_from_fname(join(folder, i))
        samples[d['name']].append(d)

    for s in samples:
        sgroup_name = samples[s][0]['samplegroup']
        if not sgroup_name in study.samplegroup_names:
            sg = RockPy3.SampleGroup(name=samples[s][0]['samplegroup'])
            study.add_samplegroup(sg)
        sg = study[sgroup_name]
        if not s in study.sdict:
            smpl = RockPy3.Sample(**samples[s][0])
            sg.add_samples(smpl)
        for m in samples[s]:
            measurement = smpl.add_measurement(**m)
            if 'ISindex' in m:
                initial = get_IS(m, samples[s])
                measurement.set_initial_state(**initial)
                samples[s].remove(initial)
            if 'IS' in m and m['IS'] == True:
                continue
    return study


def rename_file(old_file='',
                samplegroup='', sample_name='',
                mtype='', ftype='',
                mass='', mass_unit='kg',
                height='', length_unit='m',
                diameter='',
                series=None,
                std=None,
                idx=None,
                **options):
    """
    renames a file using RockPy naming convention
    """
    fname = get_fname_from_info(samplegroup=samplegroup, sample_name=sample_name,
                                mtype=mtype, ftype=ftype,
                                mass=mass, mass_unit=mass_unit,
                                height=height, length_unit=length_unit,
                                diameter=diameter,
                                series=series,
                                std=std,
                                idx=idx,
                                **options
                                )
    path = os.path.dirname(old_file)
    new_file = join(path, fname)

    def check_if_file_exists(path):
        new_path = path
        n = 1
        while os.path.exists(new_path):
            RockPy3.logger.warning('FILE {} already exists adding suffix'.format(path))
            new_path = '_'.join([path, str(n)])
            n += 1
        return new_path

    RockPy3.logger.info('RENAMING:')
    RockPy3.logger.info('{}'.format(old_file))
    RockPy3.logger.info('---->')
    RockPy3.logger.info('{}'.format(new_file))
    new_file = check_if_file_exists(new_file)
    os.rename(old_file, new_file)

class minfo():

    @staticmethod
    def extract_tuple(s):
        s = s.strip('(').strip(')').split(',')
        return tuple(s)

    def extract_series(self, s):
        print(s)
        s = self.extract_tuple(s)
        s = tuple([s[0], float(s[1]), s[2]])
        return s

    @staticmethod
    def tuple2str(tup):
        """
        takes a tuple and converts it to text, if more than one element, brackets are put around it
        """
        if tup is None:
            return ''

        if type(tup)==list:
            if len(tup) == 1:
                tup = tup[0]
            else:
                tup=tuple(tup)
        if len(tup) == 1:
            return str(tup[0])
        else:
            return str(tup).replace('\'', ' ').replace(' ','')

    def measurement_block(self, block):
        sgroups, samples, mtypes, ftype = block.split('_')
        # names with , need to be replaced
        if not '(' in samples and ',' in samples:
            samples = samples.replace(',', '.')
            RockPy3.logger.warning('sample name %s contains \',\' will be replaced with \'.\'' %samples)

        self.sgroups, self.samples, self.mtypes, self.ftype = self.extract_tuple(sgroups), self.extract_tuple(samples), self.extract_tuple(mtypes), ftype
        self.mtypes = tuple(RockPy3.abbrev_to_classname(mtype) for mtype in RockPy3._to_tuple(self.mtypes))
        self.ftype = RockPy3.abbrev_to_classname(ftype)

    def sample_block(self, block):
        out = [[None, None], [None, None], [None, None]]
        units = []

        if '_' in block:
            #old style infos
            block = block.replace('[', '').replace(']', '')
            block = block.replace(',', '.')
            parts = block.split('_')
        else:
            parts = block.split(',')

        for i in range(3):
            try:
                p = parts[i]
                val = float(re.findall(r"[-+]?\d*\.\d+|\d+", p)[0])
                unit = ''.join([i for i in p if not i.isdigit()]).strip('.')
            except IndexError:
                val = None
                unit = None
            out[i][0] = val
            out[i][1] = unit
        [self.mass, self.massunit], [self.height, self.heightunit], [self.diameter, self.diameterunit] = out

    def series_block(self, block):
        # old style series block: e.g. mtime(h)_0,0_h;mtime(m)_0,0_min;speed_1100,0_rpm
        if not any(s in block for s in ('(',')')) or ';' in block:
            block = block.replace(',', '.')
            block = block.replace('_', ',')
            block = block.replace(';', '_')

        series = block.split('_')
        if not series:
            self.series = None
        self.series =[self.extract_series(s) for s in series if s]

    def add_block(self, block):
        if block:
            self.additional = block
        else:
            self.additional = ''

    def comment_block(self,block):
        self.comment = block

    def get_measurement_block(self):
        block = deepcopy(self.storage[0])
        block[2] = [abbreviate_name(mtype).upper() for mtype in RockPy3._to_tuple(block[2]) if mtype]
        block[3] = abbreviate_name(block[3]).upper()
        if not all(block[1:]):
            raise ImportError('sname, mtype, ftype needed for minfo to be generated')
        return '_'.join((self.tuple2str(b) for b in block))

    def get_sample_block(self):
        out = ''
        block = self.storage[1]

        if not any((all(b) for b in block)):
            return None

        for i, b in enumerate(block):
            if not all(b):
                if i == 0:
                    aux = 'XXmg'
                else:
                    aux = 'XXmm'
            else:
                aux = ''.join(map(str, b))
            if not out:
                out = aux
            else:
                out = ','.join([out, aux])

            # stop if no more entries follow
            if not any(all(i) for i in block[i + 1:]):
                break
        return out

    def get_series_block(self):
        block = self.storage[2]
        if block:
            if type(block[0]) != tuple:
                block = (block,)
            out = [self.tuple2str(b) for b in block]
            return '_'.join(out)

    def get_add_block(self):
        if self.additional:
            out = tuple(''.join(map(str, a)) for a in self.additional)
            return self.tuple2str(out)

    def is_readable(self):
        if not os.path.isfile(self.fpath):
            return False
        if all(self.storage[0][1:]):
            return True
        else:
            return False

    def __init__(self, fpath,
                 sgroups=None, samples=None,
                 mtypes=None, ftype=None,
                 mass=None, height=None, diameter=None,
                 massunit=None, lengthunit=None, heightunit=None, diameterunit=None,
                 series=None, comment=None, folder=None, suffix=None,
                 read_fpath=True, **kwargs):

        """

        Parameters
        ----------
        fpath
        sgroups
        samples
        mtypes
        ftype
        mass
        height
        diameter
        massunit
        lengthunit
        heightunit
        diameterunit
        series
        comment
        folder
        suffix
        read_fpath: bool
            if true the path will be read for info
        kwargs
        """
        if 'mtype' in kwargs and not mtypes:
            mtypes = kwargs.pop('mtype')
        if 'sgroup' in kwargs and not sgroups:
            mtypes = kwargs.pop('sgroup')
        if 'sample' in kwargs and not samples:
            mtypes = kwargs.pop('sample')


        blocks = (self.measurement_block, self.sample_block, self.series_block, self.add_block, self.comment_block)
        additional = tuple()

        sgroups = RockPy3._to_tuple(sgroups)
        sgroups = tuple([sg if sg != 'None' else None for sg in sgroups])

        if mtypes:
            mtypes = tuple(RockPy3.abbrev_to_classname(mtype) for mtype in RockPy3._to_tuple(mtypes))
        if ftype:
            ftype = RockPy3.abbrev_to_classname(ftype)

        self.__dict__.update({i: None for i in ('sgroups', 'samples', 'mtypes', 'ftype',
                                               'mass', 'height', 'diameter',
                                               'massunit', 'lengthunit', 'heightunit', 'diameterunit',
                                               'series', 'additional', 'comment', 'folder', 'suffix')
                              })
        self.fpath = fpath

        if read_fpath and fpath: #todo add check for if path is readable
            self.folder= os.path.dirname(fpath)
            f, self.suffix = os.path.splitext(os.path.basename(fpath))
            self.suffix = self.suffix.strip('.')
            splits = f.split('#')

            #check if RockPy compatible e.g. first part must be len(4)
            if not len(splits[0]) == 4:
                pass
            for i, block in enumerate(blocks[:len(splits)]):
                if splits[i]:
                    try:
                        block(splits[i])
                    except (ValueError, ):
                        pass
        for i in ('sgroups', 'samples', 'mtypes', 'ftype',
                  'mass', 'height', 'diameter',
                  'massunit', 'lengthunit', 'heightunit', 'diameterunit',
                  'series', 'additional', 'comment', 'folder'):

            if locals()[i]:
                if isinstance(locals()[i], (tuple, list, set)):
                    if not all(locals()[i]):
                        continue
                setattr(self, i, locals()[i])

        if self.additional is None:
            self.additional = ''
        if kwargs:
            for k,v in kwargs.items():
                if v:
                    print(k,v, self.additional)
                    self.additional += '{}:{}'.format(k,v)

        if suffix:
            self.suffix = suffix

        if type(self.suffix)==int:
            self.suffix = '%03i'%self.suffix

        if not self.suffix:
            self.suffix = '000'

        if not self.sgroups: self.sgroups = None

        self.storage = [[self.sgroups, self.samples, self.mtypes, self.ftype],
               [[self.mass, self.massunit], [self.height, self.heightunit], [self.diameter, self.diameterunit],],
               self.series,
               (self.additional,),
               self.comment]

    @property
    def fname(self):
        """
        name after new RockPy3 convention
        """

        # if not self.fpath:
        #     RockPy3.logger.error('%s is not a file' %self.get_measurement_block())
        #     return
        out = [self.get_measurement_block(), self.get_sample_block(),
               self.get_series_block(), self.get_add_block(), self.comment]


        for i, block in enumerate(out[::-1]):
            if not block:
                out.pop()
            else:
                break
        fname = '#'.join(map(str, out))+'.'+self.suffix
        fname = fname.replace('None', '')
        return fname

    @property
    def measurement_infos(self):
        idict = {'fpath': self.fpath, 'ftype': self.ftype, 'idx': self.suffix, 'series': self.series}
        samples = RockPy3._to_tuple(self.samples)
        for i in samples:
            for j in self.mtypes:
                mtype = RockPy3.abbrev_to_classname(j)
                idict.update({'mtype':mtype, 'sample':i})
                yield idict

    @property
    def sample_infos(self):
        sdict = dict(mass=self.mass, diameter=self.diameter, height=self.height,
                     mass_unit = self.massunit, height_unit=self.heightunit, diameter_unit=self.diameterunit,
                     samplegroup=self.sgroups)

        samples = RockPy3._to_tuple(self.samples)
        for i in samples:
            sdict.update({'name': i})
            yield sdict

if __name__ == '__main__':
    m = RockPy3.Sample('test').add_measurement(fpath='/Users/mike/Dropbox/experimental_data/FeNiX/FeNi20K/separation test/FeNi_FeNi20-Ka1440_HYS_VSM###Tesa1.001')

    print(m.sobj.samplegroups)