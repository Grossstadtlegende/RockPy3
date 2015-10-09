__author__ = 'mike'
from RockPy.core import io


class Generic(io.ftype):
    def __init__(self):
        super(Generic, self).__init__(dfile=None, sample_name=None)
        self.generate = True

class Synthetic(io.ftype):
    pass