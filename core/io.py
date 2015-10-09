__author__ = 'volk'
import logging
import gc
import RockPy3
class ftype(object):

    log = logging.getLogger('RockPy.io.')

    def split_tab(self, line):
        return line.split('\t')

    @classmethod
    def get_subclass_name(cls):
        return cls.__name__

    def __init__(self, dfile, sample_name=None, dialect=None):
        """
        Constructor of the basic file type instance
        """
        # self.log = logging.getLogger('RockPy.io.'+self.get_subclass_name())
        self.log.info('IMPORTING << %s , %s >> file: << %s >>' % (sample_name, type(self).__name__, dfile))
        # ftype.log.info('IMPORTING << %s , %s >> file: << %s >>' % (sample_name, type(self).__name__, dfile))
        self.sample_name = sample_name
        self.file_name = dfile

        # initialize
        self.raw_data = None
        # self.data = None

    def simple_import(self):
        """
        simple wrapper that opens file and uses file.readlines to import and removes newline marks
        :return:
        """
        with open(self.file_name, 'r', encoding="ascii", errors="surrogateescape") as f:
            out = f.readlines()
        out = map(str.rstrip, out)
        return list(out)

    @property
    def file_header(self):
        header = []
        return header

    @property
    def float_list(self):
        list = ['x', 'y', 'z', 'm']
        return float

    @staticmethod
    def convert2float_or_str(item):
        """
        Converts an item to a float or if not possible returns the str

        Parameters
        ----------
            item: str

        Returns
        -------
            str, float
        """
        with RockPy3.ignored(ValueError):
            item = float(item)
        return item

class generic(ftype):
    pass