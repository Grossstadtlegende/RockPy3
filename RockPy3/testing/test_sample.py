from unittest import TestCase
import RockPy3

__author__ = 'mike'


class TestSample(TestCase):
    def setUp(self):
        self.study = RockPy3.Study()

    def test_init_works(self):
        s = RockPy3.Sample(name='test_sample', comment='no comment')
        self.assertTrue(s.name == 'test_sample')
        self.assertTrue(s.comment == 'no comment')

        # check if a study was created
        self.assertTrue(s in s._study.samples.keys())
        self.assertTrue(s in s._study.samplelist)
        self.assertTrue(s in s._study.samplenames)
