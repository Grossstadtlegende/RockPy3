__author__ = 'mike'
import logging
import numpy as np
import RockPy3
log = logging.getLogger('RockPy3.utils.convert')

def convert2(in_unit, out_unit, unit_type):
    """

    :param in_unit: str the data unit
    :param out_unit: str the desired output unit
    :param unit_type: what kind of units they are.
    :return: conversion factor

    :example:

        assume you want to convert a samples mass from *mg* to *kg*

        ``sample_mass = 0.00345  # sample weight in kg``

        ``sample_mass *= convert2('kg', 'mg', 'mass')  # gives new mass in mg``

        ``>> sample_mass = 3450.0``

    :implemented units:

         * 'volume' [only metric units]
         * 'pressure'
         * 'length' [only metric units]
         * 'mass' [only metric units]
         * 'area' [only metric units]

    """
    conversion_table = {'mass':
                            {'T': 1E-9,
                             'kg': 1E-6,
                             'g': 1E-3,
                             'mg': 1,
                             'mug': 1E3,
                             'ng': 1E6},
                        'length':
                            {'km': 1E-6,
                             'm': 1E-3,
                             'dm': 1E-2,
                             'cm': 1E-1,
                             'mm': 1,
                             'micron': 1E3,
                             'mum': 1E3,
                             'nm': 1E6},
                        'area':
                            {'km2': 1E-12,
                             'm2': 1E-6,
                             'dm2': 1E-4,
                             'cm2': 1E-2,
                             'mm2': 1,
                             'micron2': 1E6,
                             'nm2': 1E12},
                        'volume':
                            {'km3': 1E-18,
                             'm3': 1E-9,
                             'dm3': 1E-6,
                             'cm3': 1E-3,
                             'mm3': 1,
                             'micron3': 1E9,
                             'nm3': 1E18},
                        'pressure':
                            {'GPa': 0.001,
                             'MPa': 1.0,
                             'N/cm2': 1000.0,
                             'N/m2': 1000000.0,
                             'N/mm2': 1.0,
                             'Pa': 1000000.0,
                             'TPa': 1e-06,
                             'at': 10.197162129779,
                             'atm': 9.86923266716,
                             'bar': 10.0,
                             'cmHg': 750.063755419211,
                             'dyn/cm2': 10000000.0,
                             'hPa': 10000.0,
                             'kN/m2': 1000.0,
                             'kPa': 1000.0,
                             'kgf/cm2': 10.197162129779,
                             'kgf/m2': 101971.621297793,
                             'kgf/mm2': 0.101971621298,
                             'lbf/ft2': 20885.434233120002,
                             'lbf/in2': 145.03773773,
                             'mbar': 10000.0,
                             'mmHg': 7500.63755419211,
                             'mubar': 10000000.0,
                             'psi': 145.03773773,
                             'torr': 7500.61682704},
                        }

    if unit_type not in conversion_table:
        log.error('UNKNOWN\t << unit_type >>')

    conversion = conversion_table[unit_type]

    if in_unit not in conversion:
        log.error('UNKNOWN\t in_unit << %s, %s >>' %(in_unit, unit_type))
        return
    if out_unit not in conversion:
        log.error('UNKNOWN\t out_unit << %s, %s >>' %(in_unit, unit_type))
        return

    factor = conversion[out_unit] / conversion[in_unit]

    if not in_unit == out_unit:
        log.debug('CONVERSION %s -> %s : FACTOR = %.2e' % (in_unit, out_unit, factor))

    return factor


def get_conversion_table(unit=None):
    """
    This is a conversion table. Ask for the unit like:
    conversion['mass'] returns the appropriate conversion factor

    Parameters
    ----------
    """

    conversion_table = {'mass':
                            {'T': 1E-9,
                             'kg': 1E-6,
                             'g': 1E-3,
                             'mg': 1,
                             'mug': 1E3,
                             'ng': 1E6},
                        'length':
                            {'km': 1E-6,
                             'm': 1E-3,
                             'dm': 1E-2,
                             'cm': 1E-1,
                             'mm': 1,
                             'micron': 1E3,
                             'mum': 1E3,
                             'nm': 1E6},
                        'area':
                            {'km2': 1E-12,
                             'm2': 1E-6,
                             'dm2': 1E-4,
                             'cm2': 1E-2,
                             'mm2': 1,
                             'micron2': 1E6,
                             'nm2': 1E12},
                        'volume':
                            {'km3': 1E-18,
                             'm3': 1E-9,
                             'dm3': 1E-6,
                             'cm3': 1E-3,
                             'mm3': 1,
                             'micron3': 1E9,
                             'nm3': 1E18},
                        'pressure':
                            {'GPa': 0.001,
                             'MPa': 1.0,
                             'N/cm2': 1000.0,
                             'N/m2': 1000000.0,
                             'N/mm2': 1.0,
                             'Pa': 1000000.0,
                             'TPa': 1e-06,
                             'at': 10.197162129779,
                             'atm': 9.86923266716,
                             'bar': 10.0,
                             'cmHg': 750.063755419211,
                             'dyn/cm2': 10000000.0,
                             'hPa': 10000.0,
                             'kN/m2': 1000.0,
                             'kPa': 1000.0,
                             'kgf/cm2': 10.197162129779,
                             'kgf/m2': 101971.621297793,
                             'kgf/mm2': 0.101971621298,
                             'lbf/ft2': 20885.434233120002,
                             'lbf/in2': 145.03773773,
                             'mbar': 10000.0,
                             'mmHg': 7500.63755419211,
                             'mubar': 10000000.0,
                             'psi': 145.03773773,
                             'torr': 7500.61682704},
                        }

    if not unit or unit == 'all':
        return conversion_table
    else:
        if unit not in conversion_table:
            RockPy3.log.error('unit not recognized')
            return
        else:
            return conversion_table[unit]


def si_units(unit):
    si_table = {'mass': 'kg',
                'length': 'm',
                'area': 'm2',
                'volume': 'm3',
                'pressure': 'Pa',
                }
    if unit:
        if unit in si_table:
            return si_table[unit]
    else:
        return si_table


def lookup_unit_type(unit=None):
    """
    helper function that lookups what unittype a certain unit belongs to.

    Example:
    :return:
    """
    table = get_conversion_table()
    out = {i: k for k, v in table.iteritems() for i in v}

    if not unit:
        return out
    else:
        try:
            return out[unit]
        except KeyError:
            raise KeyError('unit << %s >> not recognized' % unit)

def get_significant_digits(values):
    values = RockPy3._to_tuple(values)
    values = map(str, values)
    digits = [len(s.split('.')[-1]) for s in values]
    return digits if len(digits)>1 else digits[0]

def get_unit_prefix(value, SI_unit):
    """
    Takes a value and checks, which is the best prefix
    """
    prefix = {-15:('femto', 'f'), -12 :('pico', 'p'), -9 : ('nano', 'n'), -6: ('micro', 'mu'), -3:('milli', 'm'), 0:('', ''),
              3:('kilo', 'k'), 6:('mega', 'M')}

    exp = np.floor(np.log10(value))  # exponent of mass
    digits = get_significant_digits(value)

    if SI_unit == 'kg':
        exp += 3
        SI_unit = 'g'

    for pexp, pref in sorted(prefix.items()):  # todo write function
        if exp-2 <= pexp:
            value /= np.power(10., pexp - 3)
            unit = pref[1] + SI_unit
            break
        else:
            unit = SI_unit

    return np.round(value,digits), unit

if __name__ == '__main__':
    print(get_significant_digits(0.11))
    print(get_unit_prefix(0.00000000000001, 'kg'))