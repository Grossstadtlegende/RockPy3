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
        self.assertTrue(m.has_calculation_method('hf_sus'))
        self.assertFalse(m.has_calculation_method('ms'))

    def test_result_category(self):
        self.fail()

    def test_get_calculate_methods(self):
        self.fail()

    def test_simulate(self):
        self.fail()

    def test_measurement_result(self):
        self.fail()

    def test_implemented_ftypes(self):
        self.fail()

    def test_subclasses(self):
        self.fail()

    def test_inheritors(self):
        self.fail()

    def test_implemented_measurements(self):
        parameters = ['height', 'diameter', 'mass', 'volume', 'locationgeo']
        self.assertTrue(all(i in RockPy3.core.measurement.Measurement.implemented_measurements() for i in parameters))

    def test_measurement_formatters(self):
        self.fail()

    def test_result_methods(self):
        print( RockPy3.core.measurement.Measurement.result_methods())
        self.fail()

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
        self.fail()

    def test_import_data(self):
        self.fail()

    def test_set_initial_state(self):
        self.fail()

    def test_has_initial_state(self):
        self.fail()

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
        self.fail()

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
