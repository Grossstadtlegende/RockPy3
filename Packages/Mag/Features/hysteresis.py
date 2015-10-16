__author__ = 'mike'
import RockPy3.Packages.Generic.Features.generic

def hysteresis(ax, mobj, **plt_props):
    df_branch(ax, mobj, **plt_props)
    uf_branch(ax, mobj, **plt_props)


def df_branch(ax, mobj, **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    ax.plot(mobj.data['down_field']['field'].v,
            mobj.data['down_field']['mag'].v,
            **plt_props)


def uf_branch(ax, mobj, **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    ax.plot(mobj.data['up_field']['field'].v,
            mobj.data['up_field']['mag'].v,
            **plt_props)


def virgin_branch(ax, mobj, **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """

    ax.plot(mobj.data['virgin']['field'].v,
            mobj.data['virgin']['mag'].v,
            **plt_props)


def irreversible(ax, mobj, **plt_props):
    irrev = mobj.get_irreversible()
    ax.plot(irrev['field'].v,
            irrev['mag'].v,
            **plt_props)


def reversible(ax, mobj, **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """

    rev = mobj.get_reversible()
    ax.plot(rev['field'].v,
            rev['mag'].v,
            **plt_props)
