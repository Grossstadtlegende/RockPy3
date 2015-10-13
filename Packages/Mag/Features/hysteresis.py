__author__ = 'mike'


def hysteresis(ax, mobj, **plt_opt):
    df_branch(ax, mobj, **plt_opt)
    uf_branch(ax, mobj, **plt_opt)


def df_branch(ax, mobj, **plt_opt):
    """
    Plots the down_field branch of a hysteresis
    """
    ax.plot(mobj.data['down_field']['field'].v,
            mobj.data['down_field']['mag'].v,
            ls=mobj.linestyle, marker='', color=mobj.color, label=mobj.label,
            **plt_opt)


def uf_branch(ax, mobj, **plt_opt):
    """
    Plots the down_field branch of a hysteresis
    """
    ax.plot(mobj.data['up_field']['field'].v,
            mobj.data['up_field']['mag'].v,
            ls=mobj.linestyle, marker='', color=mobj.color,
            **plt_opt)


def virgin_branch(ax, mobj, **plt_opt):
    """
    Plots the down_field branch of a hysteresis
    """
    ls = plt_opt.pop('ls', '-')
    marker = plt_opt.pop('marker', '')

    ax.plot(mobj.data['virgin']['field'].v,
            mobj.data['virgin']['mag'].v,
            ls=mobj.linestyle, marker='', color=mobj.color,
            **plt_opt)


def irreversible(ax, mobj, **plt_opt):
    irrev = mobj.get_irreversible()
    ax.plot(irrev['field'].v,
            irrev['mag'].v,
            ls=mobj.linestyle, marker='', color=mobj.color, label=mobj.label,
            **plt_opt)


def reversible(ax, mobj, **plt_opt):
    """
    Plots the down_field branch of a hysteresis
    """

    rev = mobj.get_reversible()
    ax.plot(rev['field'].v,
            rev['mag'].v,
            ls=mobj.linestyle, marker='', color=mobj.color, label=mobj.label,
            **plt_opt)
