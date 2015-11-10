__author__ = 'mike'

def demagnetization_data(ax, mobj, dtype='mag', **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    ax.plot(mobj.data['data']['variable'].v,
            mobj.data['data'][dtype].v,
            **plt_props)