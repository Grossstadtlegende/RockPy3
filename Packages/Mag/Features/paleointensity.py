__author__ = 'mike'

def paleointensity_data_points(ax, mobj, dtype='mag', **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    vars = sorted(list(set(mobj.data['acquisition']['variable'].v) & set(mobj.data['demagnetization']['variable'].v)))
    xdata = mobj.data['acquisition'].filter_idx([i for i,v in enumerate(mobj.data['acquisition']['variable'].v) if v in vars])
    ydata = mobj.data['demagnetization'].filter_idx([i for i,v in enumerate(mobj.data['demagnetization']['variable'].v) if v in vars])
    plt_props.update(dict(linestyle=''))
    ax.plot(xdata[dtype].v,
            ydata[dtype].v,
            **plt_props)

def paleointensity_data_line(ax, mobj, dtype='mag', **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    vars = sorted(list(set(mobj.data['acquisition']['variable'].v) & set(mobj.data['demagnetization']['variable'].v)))
    xdata = mobj.data['acquisition'].filter_idx([i for i,v in enumerate(mobj.data['acquisition']['variable'].v) if v in vars])
    ydata = mobj.data['demagnetization'].filter_idx([i for i,v in enumerate(mobj.data['demagnetization']['variable'].v) if v in vars])

    plt_props.update(dict(marker=''))
    ax.plot(xdata[dtype].v,
            ydata[dtype].v,
            **plt_props)