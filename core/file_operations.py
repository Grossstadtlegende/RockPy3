from RockPy3.utils.general import add_unit

__author__ = 'mike'
import os
import os.path
from os.path import expanduser, join
from collections import defaultdict
import numpy as np
from pint import UnitRegistry
import RockPy3
import pickle
import RockPy3.core

ureg = UnitRegistry()
default_folder = join(expanduser("~"), 'Desktop', 'RockPy')


def mtype_ftype_abbreviations():
    RockPy3.logger.debug('READING FTYPE/MTYPE abbreviations')
    with open(join(RockPy3.installation_directory, 'abbreviations')) as f:
        abbrev = f.readlines()
    abbrev = [tuple(i.rstrip().split(':')) for i in abbrev if i.rstrip() if not i.startswith('#')]
    abbrev = dict((i[0], [j.lstrip() for j in i[1].split(',')]) for i in abbrev)
    inv_abbrev = {i: k for k in abbrev for i in abbrev[k]}
    inv_abbrev.update({k:k for k in abbrev})
    return inv_abbrev, abbrev


def fname_abbrevs():
    abbrev = RockPy3.mtype_ftype_abbreviations
    out = {}
    for k, v in abbrev.iteritems():
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
        dump = cPickle.dumps(smthg)
        f.write(dump)
        f.close()

def abbrev_to_name(abbrev):
    if not abbrev:
        return ''
    return RockPy3.mtype_ftype_abbreviations_inversed[abbrev.lower()]

def name_to_abbrev(name):
    if not name:
        return ''
    return RockPy3.mtype_ftype_abbreviations[name.lower()][0]

def convert_to_fname_abbrev(name_or_abbrev):
    if name_or_abbrev:
        name = abbrev_to_name(name_or_abbrev)
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
        out = cPickle.loads(f.read())
    return out


def get_fname_from_info(samplegroup='', sample_name='',
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
    abbrev = fname_abbrevs()

    mtype = abbrev_to_name(mtype)
    ftype = abbrev_to_name(ftype)

    if not mtype.lower() in RockPy3.implemented_measurements():
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

    sample_info = '_'.join(
        [add_unit(str(mass).replace('.', ','), mass_unit),
         add_unit(str(diameter).replace('.', ','), length_unit),
         add_unit(str(height).replace('.', ','), length_unit),
         ])

    params = ';'.join(
        ['_'.join(map(str, [stypes[i], svals[i], sunits[i]])) for i in range(len(stypes))])

    if options:
        opt = ';'.join(['_'.join([k, str(v)]) for k, v in sorted(options.iteritems())])
    else:
        opt = ''
    if std:
        std = 'STD%s' % str(std)
    else:
        std = ''
    out = '#'.join([sample, sample_info, params, std, opt])
    out += '.%s' % idx
    return out


def get_info_from_fname(path=None):
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

    out = {
        'samplegroup': samplegroup,
        'name': sample_name,
        'mtype': mtype,
        'ftype': ftype,
        'fpath': join(folder, fpath),
        'series': series,
        'std': STD,
        'idx': int(index)
    }

    if mass[0]:
        out.update({'mass': mass[0],
                    'mass_unit': mass[1]})
    if diameter[0]:
        out.update({'diameter': diameter[0],
                    'length_unit': diameter[1]})
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
            new_path= '_'.join([path, str(n)])
            n += 1
        return new_path

    RockPy3.logger.info('RENAMING:')
    RockPy3.logger.info('{}'.format(old_file))
    RockPy3.logger.info('---->')
    RockPy3.logger.info('{}'.format(new_file))
    new_file = check_if_file_exists(new_file)
    os.rename(old_file, new_file)