__author__ = 'Mike'

import RockPy3
def thermocurve_data(ax, mobj, dtype='mag', **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    for dtype in mobj.data:
        if 'warming' in dtype:
            plt_props.update({'color': '#DC3522'})
        if 'cooling' in dtype:
            plt_props.update({'color': '#00305A'})
        ax.plot(mobj.data[dtype]['temp'].v,
                mobj.data[dtype]['mag'].v,
                **plt_props)


def thermocurve_data_colored(ax, mobj, dtype='mag', **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    for dtype in mobj.data:
        ax.plot(mobj.data[dtype]['temp'].v,
                mobj.data[dtype]['mag'].v,
                **plt_props)


if __name__ == '__main__':
    import RockPy3

    S = RockPy3.Study
    S.import_folder('/Users/Mike/Dropbox/experimental_data/pyrrhotite/rmp_||a')
    S.import_folder('/Users/Mike/Dropbox/experimental_data/pyrrhotite/rmp_||b')
    S.import_folder('/Users/Mike/Dropbox/experimental_data/pyrrhotite/rmp_||c')
    S.color_from_series(stype='field')
    S.label_add_series(stypes='field', add_stype=False)
    fig = RockPy3.Figure(linewidth=1.5)
    v = fig.add_visual('thermocurve', features='thermocurve_data_colored', data=S.get_measurement(stype='axis', sval=1))
    v.title = 'a-axis'
    v = fig.add_visual('thermocurve', features='thermocurve_data_colored', data=S.get_measurement(stype='axis', sval=2))
    v.title = 'b-axis'
    v = fig.add_visual('thermocurve', features='thermocurve_data_colored', data=S.get_measurement(stype='axis', sval=3))
    v.title = 'c-axis'
    f = fig.show(xlims=[10, 100], equal_lims=True, return_figure=False, save_path='Desktop', file_name='LTPY_rmp_3axis')

    # f.set_figwidth(15)
    # f.savefig('/Users/Mike/Dropbox/experimental_data/pyrrhotite/LTPY_FC_plots.pdf')
