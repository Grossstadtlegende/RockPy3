__author__ = 'mike'
import RockPy

print(RockPy.get_fname_from_info(sample_group='LF3C', sample_name='Ve', mtype='hys',
                                 ftype='vftb', mass=267.9, mass_unit='mg',
                                 series=[('temp', 0,"C")]))