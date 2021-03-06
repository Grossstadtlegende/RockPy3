import os
import logging
import matplotlib
import configparser

# matplotlib.use('Qt5Agg')

import RockPy3
installation_directory = os.path.dirname(RockPy3.__file__)

# automatic import of all subpackages in Packages and core
import pkgutil
subpackages = sorted([(i[1], i[2]) for i in pkgutil.walk_packages([os.path.dirname(RockPy3.__file__)], prefix='RockPy3.')])

for i in subpackages:
    # dont import testing packages
    if 'test' in i[0]:
        continue
    # store latest package name
    if i[1]:
        package = i[0]
        __import__(package)
    # the latest package name needs to be in the name of the 'non'-package to be imported
    if not i[1] and package in i[0]:
        #import the file in the package e.g. 'Packages.Mag.Visuals.paleointensity'
        __import__(i[0])

from RockPy3.core.visual import Visual
from RockPy3.utils.general import QuickFig

from RockPy3.core.visual import set_colorscheme

# core imports
from RockPy3.core.study import Study as RockPyStudy
from RockPy3.core.sample import Sample, MeanSample
from RockPy3.core.measurement import Measurement
from RockPy3.core.series import Series
from RockPy3.core.data import RockPyData as Data

from RockPy3.Packages.Generic.Measurements.parameters import Parameter
from RockPy3.core.figure import Figure

from RockPy3.core.utils import ignored
from RockPy3.core.utils import to_list, _to_tuple

from RockPy3.core.data import condense

from RockPy3.core.file_operations import save, load, abbrev_to_classname, minfo
from RockPy3.core.file_operations import get_fname_from_info, get_info_from_fname, import_folder
from RockPy3.core.file_operations import load_xml, save_xml
from RockPy3.core.utils import setLatex

### utility imports
from RockPy3.utils.general import check_coordinate_system

import subprocess

RockPy3.core.utils.create_logger('RockPy3')
logger = logging.getLogger('RockPy3')
logger.propagate = False

test_data_path = os.path.join(os.getcwd().split('RockPy3')[0], 'RockPy3', 'testing', 'test_data')

dependencies = ['matplotlib', 'numpy', 'scipy', 'lmfit', 'pint', 'decorator', 'tabulate', 'basemap']


def getgitrevision():
    # get github revision and store to source file
    if not os.path.isdir(os.path.join(installation_directory, ".git")):
        logger.warning(".git not found. Can't get version from git.")
        return
    try:
        p = subprocess.Popen(["git", "describe", "--tags", "--dirty", "--always"], stdout=subprocess.PIPE)
    except EnvironmentError:
        logger.warning("unable to run git.")
        return
    stdout = p.communicate()[0]
    if p.returncode != 0:
        logger.warning("git returned error code %d" % p.returncode)
        return
    rev = stdout.strip()
    logger.debug("got revision %s from git" % rev)
    return str(rev)


def storegitrevision(revision=""):
    # strore given to _version.py
    # make sure to change to right directory before
    f = open("_rp_version.py", "w")
    f.write("__version__ = '%s'\n" % revision)
    logger.debug("git revision %s written to _rp_version.py" % revision)
    f.close()


os.chdir(installation_directory)

rev = getgitrevision()
if rev == None or rev == "":
    # no valid revision from git
    # try to use revision from _version.py
    try:
        import _rp_version

        __version__ = _rp_version.__version__
        logger.debug("got version %s from _rp_version.py" % __version__)
    except ImportError:
        __version__ = 'unknown'
        logger.warning("_rp_version.py not found. Version unknown.")
else:
    __version__ = rev
    storegitrevision(rev)

logger.info('RockPy3 rocks! Git repository version: %s' % __version__)
logger.info('RockPy3 test_data_path: %s' % test_data_path)
logger.info('RockPy3 installation_directory: %s' % installation_directory)

tabulate_available = True # assuming it is installed- checked after import_check
coordinate_systems = ('core', 'geo', 'bed')

coord = 'geo'
# log error message if default coordinate system is invalid
# check_coordinate_system(coord)

import RockPy3.import_check
import import_check
import_check.check_imports()

''' add master study '''
Study = RockPy3.core.study.Study(name='MasterStudy')

########################################################################################################################
implemented_measurements = {m.__name__.lower(): m for m in Measurement.inheritors()}
implemented_visuals = RockPy3.core.visual.Visual.implemented_visuals()

print('IMPLEMENTED MEASUREMENT TYPES     : \tFTYPES')
print('---------------------------------------------------------------------------')
print('\n'.join(['\t{:<26}: \t{}'.format(m, ', '.join(obj.measurement_formatters().keys())) for m, obj in sorted(RockPy3.implemented_measurements.items())]))

mtype_ftype_abbreviations_inversed, mtype_ftype_abbreviations = RockPy3.core.file_operations.mtype_ftype_abbreviations()

colorscheme = set_colorscheme('pretty')
linestyles = ['-', '--', ':', '-.'] * 100
marker = ['.', 's', 'o', '<', '>', '^', '+', '*', ',', '1', '3', '2', '4', '8', 'D', 'H', 'd', 'h', 'p', 'v'] * 100


def set_fontsize(fontsize=16):
    matplotlib.rcParams.update({'font.size': fontsize})
    RockPy3.fontsize = fontsize

def get_fontsize():
    return matplotlib.rcParams['font.size']

set_fontsize(14)

def CreateConfigFile():
    config = configparser.ConfigParser()
    for mtype, cls in sorted(RockPy3.implemented_measurements.items()):
        for result in cls.result_methods():
            config['#'.join([mtype, result])] = {}
            print(cls.res_signature()[result]['signature'])
            # config[mtype][] = 'test'#cls.res_signature()[result]
    #         #         if result == 'b_anc':
    #         #             print(cls.res_signature()[result])
    #         #         if not cls.res_signature()[result]['indirect']:
    #         #             standard_method = '_'.join([result, cls.result_recipe()[result]]).replace('_DEFAULT', '')
    #         #             for param, value in cls.calc_signature()[standard_method].items():
    #         #                 line = ', '.join([mtype, result, cls.result_recipe()[result].lower(), param, str(value), '\n'])
    #         #                 f.write(line)
    #
    with open(os.path.join(RockPy3.installation_directory, 'configfile.ini'), 'w') as configfile:
        config.write(configfile)


if not os.path.isfile(os.path.join(RockPy3.installation_directory, 'configfile.ini')):
    print('No config file found. Creating: %s' % os.path.join(RockPy3.installation_directory, 'configfile.ini'))
    CreateConfigFile()

if __name__ == '__main__':
    CreateConfigFile()