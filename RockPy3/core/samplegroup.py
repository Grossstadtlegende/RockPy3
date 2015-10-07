__author__ = 'mike'
import logging
from datetime import datetime
import RockPy3.core
from RockPy3.core import utils

class SampleGroup(object):
    """
    Container for Samples, has special calculation methods
    """
    log = logging.getLogger('RockPy.SampleGroup')

    count = 0

    def __init__(self, name=None):
        SampleGroup.count += 1
        SampleGroup.log.info('CRATING new << samplegroup >>')

        # ## initialize
        if name is None:
            name = 'SG%02i' % (self.count)

        self.name = name
        self.study = None  # study to which this samplegroup belongs to
        self.creation_time = datetime.now()

        self.samples = []