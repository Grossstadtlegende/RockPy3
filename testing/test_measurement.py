from unittest import TestCase
import RockPy3.core.measurement
import RockPy3.core.study

__author__ = 'mike'


class TestMeasurement(TestCase):
    # def setUp(self):
    #     self.study = RockPy3.core.study.Study()
    #     self.s = self.study.add_sample(name='s1')
    #     self.m = RockPy3.core.measurement.Measurement(sobj=self.s)

    def test_mtype_calculation_parameter_list(self):
        self.fail()

    def test_mtype_calculation_parameter(self):
        self.fail()

    def test_method_calculation_parameter_list(self):
        self.fail()

    def test_global_calculation_parameter(self):
        self.fail()

    def test_possible_calculation_parameter(self):
        self.fail()

    def test_has_recipe(self):
        import RockPy3.Packages.Mag.Measurements.hysteresis
        s = RockPy3.Sample()
        m = RockPy3.Packages.Mag.Measurements.hysteresis.Hysteresis.from_simulation(sobj=s)
        self.assertTrue(m.has_recipe('ms'))
        self.assertFalse(m.has_recipe('mrs'))

    def test_has_calculation_method(self):
        import RockPy3.Packages.Mag.Measurements.hysteresis
        s = RockPy3.Sample()
        m = RockPy3.Packages.Mag.Measurements.hysteresis.Hysteresis.from_simulation(sobj=s)
        self.assertTrue(m.has_master_method('hf_sus'))
        self.assertFalse(m.has_master_method('ms'))

    def test_has_secondary(self):
        import RockPy3.Packages.Mag.Measurements.hysteresis
        s = RockPy3.Sample()
        m = RockPy3.Packages.Mag.Measurements.hysteresis.Hysteresis.from_simulation(sobj=s)
        self.assertTrue(m.has_secondary('bcr_bc'))
        self.assertFalse(m.has_secondary('ms'))

    def test_result_category(self):
        import RockPy3.Packages.Mag.Measurements.hysteresis
        s = RockPy3.Sample()
        m = RockPy3.Packages.Mag.Measurements.hysteresis.Hysteresis.from_simulation(sobj=s)
        self.assertEqual('direct', m.result_category('mrs'))
        self.assertEqual('direct_dependent', m.result_category('bcr_bc'))
        self.assertEqual('direct_recipe', m.result_category('ms'))
        self.assertEqual('indirect_recipe', m.result_category('hf_sus'))
        # self.assertEqual('indirect_recipe', m.result_category('hf_sus'))#todo add the direct_recipe

    def test_implemented_ftypes(self):
        self.fail()

    def test_subclasses(self):
        self.fail()

    def test_inheritors(self):
        self.fail()

    def test_implemented_measurements(self):
        parameters = ['height', 'diameter', 'mass', 'volume', 'locationgeo', 'hysteresis']
        self.assertTrue(all(i in RockPy3.implemented_measurements for i in parameters))

    def test_measurement_formatters(self):
        self.fail()

    def test_result_methods(self):
        import RockPy3.Packages.Mag.Measurements.hysteresis
        s = RockPy3.Sample()
        m = RockPy3.Packages.Mag.Measurements.hysteresis.Hysteresis.from_simulation(sobj=s)
        results = sorted(m.result_methods().keys())
        self.assertAlmostEqual(['bc', 'bcr_bc', 'e_delta_t', 'e_hys', 'hf_sus', 'm_b', 'mrs', 'mrs_ms', 'ms'], results)

    def test_calculate_methods(self):
        self.fail()

    def test_correct_methods(self):
        self.fail()

    def test_get_subclass_name(self):
        self.fail()

    def test_study(self):
        self.fail()

    def test_plt_props(self):
        self.fail()

    def test_set_plt_prop(self):
        self.fail()

    def test_coord(self):
        self.fail()

    def test_stype_sval_tuples(self):
        self.fail()

    def test_m_idx(self):
        self.fail()

    def test_fname(self):
        s = RockPy3.Sample()
        s.add_measurement(
            fpath='/Users/mike/Google Drive/RockPy3/testing/test_data/FeNi_FeNi20-Jz000\'-G03_HYS_VSM#50,3[mg]_[]_[]##STD020.003')

    def test_import_data(self):
        self.fail()

    def test_set_initial_state(self):
        # only initial states
        study = RockPy3.core.study.Study()
        s = study.add_sample(name='test')
        mean = RockPy3.Study.add_sample(name='mean')
        m0 = s.add_simulation(mtype='hysteresis', noise=5)
        m1 = s.add_simulation(mtype='hysteresis', noise=5)
        mi = m1.set_initial_state(mobj=m0)
        # no deepcopy -> m0 == mi
        self.assertEqual(m0, mi)
        self.assertEqual([m0, m1], s.get_measurement(mtype='hysteresis'))
        self.assertEqual(2, len(s.measurements))
        self.assertFalse(s.measurements[0].has_initial_state)
        self.assertTrue(s.measurements[1].has_initial_state)

    def test_has_initial_state(self):
        # only initial states
        study = RockPy3.core.study.Study()
        s = study.add_sample(name='test')
        mean = RockPy3.Study.add_sample(name='mean')
        m0 = s.add_simulation(mtype='hysteresis', noise=5)
        m1 = s.add_simulation(mtype='hysteresis', noise=5)
        mi = m1.set_initial_state(mobj=m0)
        self.assertFalse(s.measurements[0].has_initial_state)
        self.assertTrue(s.measurements[1].has_initial_state)

    def test_info_dict(self):
        self.fail()

    def test__populate_info_dict(self):
        self.fail()

    def test_add_s2_info_dict(self):
        self.fail()

    def test_remove_s_from_info_dict(self):
        self.fail()

    def test_remove_all_series(self):
        self.fail()

    def test__info_dict_cleanup(self):
        self.fail()

    def test_stypes(self):
        self.fail()

    def test_svals(self):
        self.fail()

    def test_data(self):
        self.fail()

    def test_transform_data_coord(self):
        self.fail()

    def test_correction(self):
        self.fail()

    def test_reset_data(self):
        self.fail()

    def test_calculate_result(self):
        self.fail()

    def test_calc_generic(self):
        self.fail()

    def test_calc_result(self):
        self.fail()

    def test_calc_all(self):
        self.fail()

    def test_compare_parameters(self):
        self.fail()

    def test_delete_dtype_var_val(self):
        self.fail()

    def test_check_parameters(self):
        self.fail()

    def test_has_result(self):
        self.fail()

    def test_series(self):
        self.fail()

    def test__add_series_from_opt(self):
        self.fail()

    def test_has_series(self):
        self.fail()

    def test_get_series(self):
        study = RockPy3.core.study.Study()
        s = study.add_sample(name='test')
        m0 = s.add_simulation(mtype='hysteresis', noise=5, series=[('s1', 1, 'A'), ('s2', 1, 'b')])
        m1 = s.add_simulation(mtype='hysteresis', noise=5, series=[('s1', 3, 'A'), ('s2', 4, 'b')])
        self.assertEqual([1], [s.sval for s in m0.get_series(stype='s1')])
        self.assertEqual([3], [s.sval for s in m1.get_series(stype='s1')])
        self.assertEqual(['s1', 's2'], [s.stype for s in m0.get_series(sval=1)])
        self.assertEqual(['s1'], [s.stype for s in m1.get_series(sval=3)])
        self.assertEqual(['s1'], [s.stype for s in m1.get_series(series=[('s1', 3), ('s2', 1)])])

    def test_get_sval(self):
        self.fail()

    def test_add_sval(self):
        self.fail()

    def test_add_series(self):
        self.fail()

    def test__add_sval_to_data(self):
        self.fail()

    def test__add_sval_to_results(self):
        self.fail()

    def test__get_idx_dtype_var_val(self):
        self.fail()

    def test_equal_series(self):
        self.fail()

    def test_normalize(self):
        self.fail()

    def test__get_norm_factor(self):
        self.fail()

    def test__norm_method(self):
        self.fail()

    def test_get_mtype_prior_to(self):
        self.fail()

    def test__add_stype_to_results(self):
        self.fail()

    def test_get_series_labels(self):
        self.fail()

    def test_has_mtype_stype_sval(self):
        self.fail()

    def test_correct_dtype(self):
        self.fail()

    def test_set_calibration_measurement(self):
        self.fail()

    def test_label_add_sample_name(self):
        self.fail()

    def test_label_add_sample_group_name(self):
        self.fail()

    def test_label_add_study_name(self):
        self.fail()

    def test_label_add_stype(self):
        self.fail()

    def test_plottable(self):
        self.fail()

    def test_show_plots(self):
        self.fail()

    def test_set_get_attr(self):
        self.fail()

    def test_series_to_color(self):
        self.fail()

    def test_plt_all(self):
        self.fail()

    def test_report(self):
        self.fail()

    def test_etree(self):
        self.fail()

    def test_from_measurements(self):
        # 1. check that if there is no noise the data is the same
        study = RockPy3.core.study.Study()
        s = study.add_sample(name='test')
        m0 = s.add_simulation(mtype='hysteresis')  # simulation with no noise
        m1 = s.add_simulation(mtype='hysteresis')  # simulation with no noise

        mean = RockPy3.Study.add_sample(name='mean')
        mm = RockPy3.Packages.Mag.Measurements.hysteresis.Hysteresis.from_measurements_create_mean(mean, s.measurements)
        for other in s.measurements:
            self.assertTrue(all(mm.data['down_field']['mag'].v == other.data['down_field']['mag'].v))
            self.assertTrue(all(mm.data['up_field']['mag'].v == other.data['up_field']['mag'].v))


    def test_set_recipe(self):
        study = RockPy3.core.study.Study()
        s = study.add_sample(name='test')
        m0 = s.add_simulation(mtype='hysteresis')
        m0.result_bc(no_points=8, check=False)
        print(m0.results)
        # direct
        self.assertEqual('direct', m0.result_category('mrs'))
        m0.set_recipe(res='mrs', recipe='nonlinear')
        self.assertEqual('default'.upper(), m0.result_recipe['mrs'])
        # direct_recipe
        self.assertEqual('direct_recipe', m0.result_category('bc'))
        m0.set_recipe(res='bc', recipe='nonlinear')
        self.assertEqual('nonlinear'.upper(), m0.result_recipe['bc'])
        print(m0.results)
        # indirect

        # indirect_recipe
        self.assertEqual('indirect_recipe', m0.result_category('hf_sus'))
        m0.set_recipe(res='hf_sus', recipe='app2sat')
        self.assertEqual('app2sat'.upper(), m0.result_recipe['ms'])
        self.assertEqual('app2sat'.upper(), m0.result_recipe['hf_sus'])
