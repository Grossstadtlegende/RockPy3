__author__ = 'mike'
import RockPy

study = RockPy.Study()
study.import_folder('Packages/PaleoMag/testing')

m0 = study[0][0].get_measurements(svals=0)[0]
m1 = study[0][0].get_measurements(svals=0.6)[0]
m2 = study[0][0].get_measurements(svals=1.2)[0]
m3 = study[0][0].get_measurements(svals=1.8)[0]
# m3._get_ck_data()

# print m0.result_delta_ck()
fig = RockPy.Figure()
t1 = fig.add_visual(visual='arai', plt_input=m0)
t1 = fig.add_visual(visual='arai', plt_input=m1)
t1 = fig.add_visual(visual='arai', plt_input=m2)
t1 = fig.add_visual(visual='arai', plt_input=m3)

fig.show()
