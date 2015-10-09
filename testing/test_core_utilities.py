from unittest import TestCase
import RockPy3.core.file_operations
__author__ = 'mike'


class TestMtype_ftype_abbreviations(TestCase):
    def test_mtype_ftype_abbreviations(self):
        iab, ab = RockPy3.core.file_operations.mtype_ftype_abbreviations()
        self.assertEqual(['sush'], ab['sushibar'])
        self.assertEqual('sushibar', iab['sush'])
