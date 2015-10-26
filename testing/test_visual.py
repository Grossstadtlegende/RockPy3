from unittest import TestCase
import RockPy3

__author__ = 'Mike'


class TestVisual(TestCase):
    def test__add_input_to_plot(self):
        # only samples without mean
        NoMeanStudy = RockPy3.RockPyStudy(name='NoMeanStudy')
        # only samples with mean
        MeanStudy = RockPy3.RockPyStudy(name='MeanStudy')
        # samples with mean and without
        MixedStudy = RockPy3.RockPyStudy(name='MixedStudy')
        #
        # sample without mean
        NoMeanSample = RockPy3.Sample()
        Measurement_NoMeanSample = NoMeanSample.add_simulation(mtype='hys')
        # # sample with mean
        MeanSample = RockPy3.Sample()
        Measurement_MeanSample1 = MeanSample.add_simulation(mtype='hys')
        Measurement_MeanSample2 = MeanSample.add_simulation(mtype='hys')
        MeanMeasurement12 = MeanSample.create_mean_measurement(mtype='hys')
        #
        NoMeanStudy.add_sample(sobj=NoMeanSample)
        MeanStudy.add_sample(sobj=MeanSample)
        MixedStudy.add_sample(sobj=NoMeanSample)
        MixedStudy.add_sample(sobj=MeanSample)

        SampleList = [NoMeanSample, MeanSample]
        NoMeanList = [Measurement_NoMeanSample, Measurement_MeanSample1]
        MeanList = [Measurement_NoMeanSample, Measurement_MeanSample1, MeanMeasurement12]
        for i, test in enumerate([NoMeanStudy,
                                  MeanStudy, MixedStudy, #NoMeanSample, MeanSample,
                                  # Measurement_NoMeanSample, Measurement_MeanSample1,
                                  # SampleList, NoMeanList, MeanList
                                  ]):
            fig = RockPy3.Figure()
            v = fig.add_visual('hysteresis')

            # print('plot_mean=True, plot_base=True')
            mlist = v._add_input_to_plot(test, plot_mean=True, plot_base=True, base_alpha=0.1)
            # print('plot_mean=False, plot_base=True')
            mlist_NoMean = v._add_input_to_plot(test, plot_mean=False, plot_base=True, base_alpha=0.2)
            # print('plot_mean=True, plot_base=False')
            mlist_NoBase = v._add_input_to_plot(test, plot_mean=True, plot_base=False, base_alpha=0.3)
            # print('plot_mean=False, plot_base=False')
            mlist_NoMeanNoBase = v._add_input_to_plot(test, plot_mean=False, plot_base=False, base_alpha=0.4)
            # print('plot_mean=False, plot_base=False, plot_other=False')
            mlist_Nothing = v._add_input_to_plot(test, plot_mean=False, plot_base=False, plot_other=False)

            if i == 0: #NoMeanStudy
                self.assertEquals([Measurement_NoMeanSample], mlist)
                self.assertEquals([Measurement_NoMeanSample], mlist_NoMean)
                self.assertEquals([Measurement_NoMeanSample], mlist_NoBase)
                self.assertEquals([Measurement_NoMeanSample], mlist_NoMeanNoBase)
                self.assertEquals([], mlist_Nothing)
                # alpha_prop = [m._plt_props['alpha'] for m in mlist_NoMean]
                # self.assertFalse(all(i == 0.1 for i in alpha_prop))
            if i == 1: #MeanStudy
                self.assertEquals([MeanMeasurement12, Measurement_MeanSample1, Measurement_MeanSample2], mlist)
                self.assertEquals([Measurement_MeanSample1, Measurement_MeanSample2], mlist_NoMean)
                self.assertEquals([MeanMeasurement12], mlist_NoBase)
                self.assertEquals([], mlist_NoMeanNoBase)
                alpha_prop = [m._plt_props['alpha'] for m in mlist_NoMean if 'alpha' in m._plt_props]
                self.assertEqual([0.1, 0.1], alpha_prop)
            if i == 2: #MixedStudy
                self.assertEquals([MeanMeasurement12, Measurement_MeanSample1, Measurement_MeanSample2, Measurement_NoMeanSample], mlist)
                self.assertEquals([Measurement_MeanSample1, Measurement_MeanSample2, Measurement_NoMeanSample], mlist_NoMean)
                self.assertEquals([MeanMeasurement12, Measurement_NoMeanSample], mlist_NoBase)
                self.assertEquals([Measurement_NoMeanSample], mlist_NoMeanNoBase)
                alpha_prop = [m._plt_props['alpha'] for m in mlist_NoMean if 'alpha' in m._plt_props]
                self.assertEqual([0.1, 0.1], alpha_prop)

            # if i == 3: #NoMeanSample
            #     self.assertEquals([Measurement_NoMeanSample], mlist)
            # if i == 4: #MeanSample
            #     self.assertEquals([Measurement_MeanSample1, Measurement_MeanSample2, MeanMeasurement12], mlist)
            # if i == 5: #Measurement_NoMeanSample
            #     self.assertEquals([Measurement_MeanSample1, Measurement_MeanSample2, MeanMeasurement12], mlist)
            # if i == 6: #Measurement_MeanSample1
            #     self.assertEquals([Measurement_MeanSample1, Measurement_MeanSample2, MeanMeasurement12], mlist)
            # if i == 7: #SampleList
            #     self.assertEquals([Measurement_MeanSample1, Measurement_MeanSample2, MeanMeasurement12], mlist)
            # if i == 8: #NoMeanList
            #     self.assertEquals([Measurement_MeanSample1, Measurement_MeanSample2, MeanMeasurement12], mlist)
            # if i == 9: #MeanList
            #     self.assertEquals([Measurement_MeanSample1, Measurement_MeanSample2, MeanMeasurement12], mlist)
