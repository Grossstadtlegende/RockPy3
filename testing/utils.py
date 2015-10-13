from unittest import TestCase
from RockPy3.core.utils import to_list
__author__ = 'mike'


class TestTo_list(TestCase):
    def test_to_list(self):
        a = 'test'
        self.assertEqual(['test'], to_list(a))
