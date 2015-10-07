__author__ = 'mike'
import time

class Study(object):
    """
    comprises data of a whole study
    i.e. container for samplegroups
    """

    def __init__(self, name=None):
        """
        general Container for Samplegroups and Samples.

        Parameters
        ----------
            name: str
                default: 'study'
                name of the Study
        """
        if not name:
            name = time.strftime("%Y%m%d:%H%M")
        # self.log = log  # logging.getLogger('RockPy.' + type(self).__name__)
        self.name = name
        self.samples = dict() #{'sname':'sobj'}
        self._samplegroups = dict() #{'sgname':'sgobj'}
        self._all_samplegroup = None

    def __repr__(self):
        return self.name

    ####################################################################################################################
    ''' add functions '''
    def add_samplegroup(self):
        pass
    def add_mean_samplegroup(self):
        pass

    ####################################################################################################################
    ''' remove functions '''
    def remove_samplegroup(self):
        pass

    ####################################################################################################################
    ''' get functions '''

    def get_samplegroup(self):
        pass

    def get_sample(self):
        pass

    def get_measurement(self):
        pass