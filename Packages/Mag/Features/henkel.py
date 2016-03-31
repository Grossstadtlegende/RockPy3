def one_to_one_line(ax, **plt_opt):
    """
    Plots the down_field branch of a hysteresis
    """
    ls = plt_opt.pop('ls', '--')
    color = plt_opt.pop('color', '#808080')

    ax.plot([1,-1], [0, 1],
                   ls = ls,
                   color = color,
                   ** plt_opt)


def henkel_data(ax, mobj, reference='coe', **plt_opt):
    """
    Plots the down_field branch of a hysteresis
    """
    coe, irm = mobj

    if reference == 'coe':
        irm = irm.data['data'].interpolate(-coe.data['data']['variable'].v)
        coe = coe.data['data']
    if reference == 'irm':
        coe = coe.data['data'].interpolate(-irm.data['data']['variable'].v)
        irm = irm.data['data']
    if reference == 'none':
        coe = coe.data['data']
        irm = irm.data['data']

    ax.plot(coe['mag'].v/max(coe['mag'].v), irm['mag'].v/max(irm['mag'].v), **plt_opt)
