__author__ = 'mike'
import logging

from RockPy3.core.utils import create_logger
import numpy as np


class Series(object):
    create_logger('RockPy.series')

    def __init__(self, stype, value, unit):
        stype = stype.lower()
        self.data = (stype, value, unit)
        self.stype = stype
        self.value = float(value)
        self.unit = unit

    @classmethod
    def from_tuple(cls, series):
        return cls(stype=series[0], value=series[1], unit=series[2])

    def get_tuple(self):
        return  (self.stype, self.value, self.unit)

    @property
    def label(self):
        return '%.2f [%s]' % (self.value, self.unit)

    def __repr__(self):
        return '<RockPy.series> %s, %.2f, [%s]' % (self.stype, self.value, self.unit)

    def __str__(self):
        return '%s: %.2f[%s]' % (self.stype, self.value, self.unit)

    def __eq__(self, other):

        if not type(self) == type(other):
            return False
        if self.stype == other.stype:
            if np.isnan(self.value) and np.isnan(other.value):
                return True
            elif self.value == other.value:
                if self.unit == other.unit:
                    return True
            else:
                return False
        else:
            return False

    @property
    def sval(self):
        return self.value

    @property
    def v(self):
        return self.value

    @property
    def u(self):
        return self.unit

    @property
    def stype_sval_tuple(self):
        return self.data[:-1]