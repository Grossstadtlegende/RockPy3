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
        s1 = self.s.add_sample(name='s1')
        m1 = s1.add_simulation('hysteresis', series=[('a', 1, 'a'), ('b', 1, 'b'), ('c', 8, 'c')])
        m2 = s1.add_simulation('hysteresis', series=[('a', 1, 'a'), ('b', 2, 'b'), ('c', 7, 'c')])
        m3 = s1.add_simulation('hysteresis', series=[('a', 1, 'a'), ('b', 3, 'b'), ('c', 6, 'c')])
        m4 = s1.add_simulation('hysteresis', series=[('a', 1, 'a'), ('b', 4, 'b'), ('c', 5, 'c')])
        m5 = s1.add_simulation('hysteresis', series=[('a', 1, 'a'), ('b', 5, 'b'), ('c', 4, 'c')])
        m6 = s1.add_simulation('hysteresis', series=[('d', 2, 'a'), ('b', 6, 'b'), ('c', 10, 'c')])
        m7 = s1.add_simulation('hysteresis', series=[('d', 1, 'a'), ('b', 7, 'b'), ('c', 2, 'c')])

        self.assertListEqual([m1, m2, m3, m4, m5, m6, m7], s1.get_measurement())
        self.assertListEqual([m1, m2, m3, m4, m5, m6, m7], s1.get_measurement(mtype='hys'))
        self.assertListEqual([m1, m2, m3, m4, m5, m7], s1.get_measurement(sval=1))
        self.assertListEqual([m1, m2, m3, m4, m5, m6, m7], s1.get_measurement(sval=(1, 2)))
        self.assertListEqual([m2, m3, m6, m7], s1.get_measurement(sval=(2, 6)))
        self.assertListEqual([m1, m2, m3, m4, m5], s1.get_measurement(stype='a'))
        self.assertListEqual([m6, m7], s1.get_measurement(stype='d'))
        self.assertListEqual([m6, m7], s1.get_measurement(stype=('d', 'non_existent')))
        self.assertListEqual([m7], s1.get_measurement(stype=('d', 'non_existent'), sval=1))
        self.assertListEqual([m6, m7], s1.get_measurement(stype=('d', 'non_existent'), sval=(7,10)))
        self.assertListEqual([m6], s1.get_measurement(sval_range='>9'))
        self.assertListEqual([m1, m2, m3, m4, m5, m6, m7], s1.get_measurement(sval_range='<=8'))
        self.assertListEqual([m1, m2, m3, m4, m5, m7], s1.get_measurement(sval_range='1-1.4'))

    def test_info(self):
        self.fail()

    def test_add_samplegroup(self):
        s1 = self.s.add_sample(name='s1')
        s2 = self.s.add_sample(name='s2')
        s3 = self.s.add_sample(name='s3')

        self.assertEqual([s1, s2], sorted(self.s.add_samplegroup(gname='A', sname=['s1', 's2'])))
