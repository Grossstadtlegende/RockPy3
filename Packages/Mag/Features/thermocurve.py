__author__ = 'Mike'


def thermocurve_data(ax, mobj, dtype='mag', **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    for dtype in mobj.data:
        if 'warming' in dtype:
            plt_props.update({'color':'r'})
        if 'cooling' in dtype:
            plt_props.update({'color':'b'})
        ax.plot(mobj.data[dtype]['temp'].v,
                mobj.data[dtype]['mag'].v,
                **plt_props)
