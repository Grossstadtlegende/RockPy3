from unittest import TestCase
import RockPy3

__author__ = 'mike'


class TestStudy(TestCase):
    def setUp(self):
        self.s = RockPy3.Study()

    def test_add_samplegroup(self):
        # name given
        sg = self.s.add_samplegroup(name='test')
        self.assertIsInstance(sg, RockPy3.SampleGroup)
        self.assertTrue(sg.name in self.s._samplegroups)
        self.assertTrue(sg.name in self.s.groupnames)
        self.assertTrue(sg in self.s.grouplist)
        # no name given
        sg = self.s.add_samplegroup()
        self.assertIsInstance(sg, RockPy3.SampleGroup)
        self.assertTrue(sg.name in self.s._samplegroups)
        self.assertTrue(sg.name in self.s.groupnames)
        self.assertTrue(sg in self.s.grouplist)

    def test_add_mean_samplegroup(self):
        self.fail()

    def test_remove_samplegroup(self):
        self.fail()

    def test_get_samplegroup(self):
        self.fail()

    def test_get_sample(self):
        self.fail()

    def test_get_measurement(self):
        self.fail()

    def test_info(self):
        self.fail()
