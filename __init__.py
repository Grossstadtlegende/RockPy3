# import os
# import logging
# import matplotlib
# # matplotlib.use('QT4Agg') #not working on OSX!
#
# import RockPy3.core.utils
# RockPy3.core.utils.create_logger('RockPy3')
# logger = logging.getLogger('RockPy3')
#
# import RockPy3.core
# # import RockPy3.core.study
# # from RockPy3.core.sample import Sample#, MeanSample
# # # from RockPy3.core.data import RockPy3Data, condense
# # # from RockPy3.core.measurement import Measurement
# # # from RockPy3.core.series import Generic as Series
# # # from RockPy3.core.file_operations import save, load, abbrev_to_name
# # # from RockPy3.core.file_operations import get_fname_from_info, get_info_from_fname, import_folder
#
# # # from RockPy3.utils.general import ignored, check_coordinate_system
#
# # from RockPy3.Visualize import Figure
# # from RockPy3.Packages import *
# # import RockPy3.Packages
# import pkgutil
# import subprocess
#
# test_data_path = os.path.join(os.getcwd().split('RockPy3')[0], 'RockPy3', 'Tutorials', 'test_data')
# installation_directory = os.path.dirname(RockPy3.__file__)
# #
# def getgitrevision():
#     # get github revision and store to source file
#     if not os.path.isdir(os.path.join(installation_directory, ".git")):
#         logger.warning(".git not found. Can't get version from git.")
#         return
#     try:
#         p = subprocess.Popen(["git", "describe", "--tags", "--dirty", "--always"], stdout=subprocess.PIPE)
#     except EnvironmentError:
#         logger.warning("unable to run git.")
#         return
#     stdout = p.communicate()[0]
#     if p.returncode != 0:
#         logger.warning("git returned error code %d" % p.returncode)
#         return
#     rev = stdout.strip()
#     logger.debug("got revision %s from git" % rev)
#     return rev
#
# def storegitrevision( revision=""):
#     # strore given to _version.py
#     # make sure to change to right directory before
#     f = open("_rp_version.py", "w")
#     f.write("__version__ = '%s'\n" % revision)
#     logger.debug("git revision %s written to _rp_version.py" % revision)
#     f.close()
#
# os.chdir(installation_directory)
#
# rev = getgitrevision()
# if rev == None or rev == "":
#     # no valid revision from git
#     # try to use revision from _version.py
#     try:
#         import _rp_version
#         __version__ = _rp_version.__version__
#         logger.debug("got version %s from _rp_version.py" % __version__)
#     except ImportError:
#         __version__ = 'unknown'
#         logger.warning("_rp_version.py not found. Version unknown.")
# else:
#     __version__ = rev
#     storegitrevision(rev)
#
#
# logger.info('RockPy3 rocks! Git repository version: %s' % __version__)
# logger.info('RockPy3 test_data_path: %s' % test_data_path)
# logger.info('RockPy3 installation_directory: %s' % installation_directory)
#
#
# logger.info('using matplotlib version %s' % matplotlib.__version__)
#
# coordinate_systems = ('core', 'geo', 'bed')
#
# coord = 'geo'
# # log error message if default coordinate system is invalid
# # check_coordinate_system(coord)
#
#
# import numpy
# logger.info('using numpy version %s' % numpy.__version__)
#
# # CoreStudy = RockPy3.core.study.Study
# # mtype_ftype_abbreviations_inversed, mtype_ftype_abbreviations = RockPy3.core.file_operations.mtype_ftype_abbreviations()
