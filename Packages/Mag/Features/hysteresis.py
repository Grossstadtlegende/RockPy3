__author__ = 'mike'
import RockPy3.Packages.Generic.Features.generic


def hysteresis(ax, mobj, **plt_props):
    df_branch(ax, mobj, **plt_props)
    # remove label otehrwise plottes twice
    plt_props.pop('label', None)
    uf_branch(ax, mobj, **plt_props)

def hysteresis_derivative(ax, mobj, **plt_props):
    for branch in ('down_field', 'up_field'):
        data = mobj.data[branch].derivative('mag', 'field')
        if branch == 'up_field':
            plt_props.pop('label', None)

        ax.plot(data['field'].v,
            data['mag'].v,
            **plt_props)

def hysteresis_error(ax, mobj, **plt_props):
    plt_props.setdefault('marker', '')
    plt_props['marker'] = ''
    plt_props.setdefault('linestyle', '')
    plt_props['linestyle'] = ''
    df_branch_error(ax, mobj, **plt_props)
    uf_branch_error(ax, mobj, **plt_props)


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


def df_branch_error(ax, mobj, **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """

    ax.errorbar(
        x=mobj.data['down_field']['field'].v,
        y=mobj.data['down_field']['mag'].v,
        xerr=mobj.data['down_field']['field'].e,
        yerr=mobj.data['down_field']['mag'].e,
        **plt_props)

def uf_branch_error(ax, mobj, **plt_props):
    """
    Plots the up_field branch of a hysteresis
    """
    ax.errorbar(
        x=mobj.data['up_field']['field'].v,
        y=mobj.data['up_field']['mag'].v,
        xerr=mobj.data['up_field']['field'].e,
        yerr=mobj.data['up_field']['mag'].e,
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
