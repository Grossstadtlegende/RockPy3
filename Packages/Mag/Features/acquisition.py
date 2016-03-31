__author__ = 'mike'

def acquisition_data(ax, mobj, dtype='mag', **plt_props):
    """
    Plots the dtype data of an acquisition measurement

    """
    ax.plot(mobj.data['data']['variable'].v,
            mobj.data['data'][dtype].v,
            **plt_props)

def cumsum_acquisition_data(ax, mobj, dtype='mag', **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    ax.plot(mobj.cumulative['variable'].v,
            mobj.cumulative[dtype].v,
            **plt_props)