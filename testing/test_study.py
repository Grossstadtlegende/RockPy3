from unittest import TestCase

import RockPy3

__author__ = 'mike'
import RockPy3.core.study


class TestStudy(TestCase):
    def setUp(self):
        self.s = RockPy3.core.study.Study('TestStudy')

    def test_add_sample(self):
        # name given
        s = self.s.add_sample(name='test')
        self.assertIsInstance(s, RockPy3.Sample)
        self.assertTrue(s.name in self.s._samples)
        self.assertTrue(s == self.s._samples[s.name])
        self.assertTrue(s.name in self.s.samplenames)
        self.assertTrue(s in self.s.samplelist)

    def test_groupnames(self):
        gtest = ['A', 'B']
        s = self.s.add_sample(name='test', samplegroup=gtest)
        self.assertEqual(gtest, self.s.groupnames)

    def test_add_mean_samplegroup(self):
        self.fail()

    def test_remove_samplegroup(self):
        gtest = ['A', 'B']
        s = self.s.add_sample(name='test', samplegroup=gtest)
        self.s.remove_samplegroup(gname='B')
        self.assertEqual(['A'], self.s.groupnames)

    def test_get_samplegroup(self):
        gtest = ['A', 'B']
        s1 = self.s.add_sample(name='s1', samplegroup=gtest)
        s2 = self.s.add_sample(name='s2', samplegroup='A')
        s3 = self.s.add_sample(name='s3', samplegroup='B')
        self.assertEqual(sorted([s1, s2]), sorted(self.s.get_samplegroup(gname='A')))


    def test_get_sample(self):
        gtest = ['A', 'B']
        s1 = self.s.add_sample(name='s1', samplegroup=gtest)
        s2 = self.s.add_sample(name='s2', samplegroup='A')
        s3 = self.s.add_sample(name='s3', samplegroup='B')

        self.assertEqual(sorted([s1, s2]), sorted(self.s.get_sample(gname='A')))

    def test_get_measurement(self):
        self.fail()

    def test_info(self):
        self.fail()
