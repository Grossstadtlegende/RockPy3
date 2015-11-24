__author__ = 'mike'
import RockPy3.Packages.Generic.Features.generic


def backfield_data(ax, mobj, **plt_props):
    ax.plot(mobj.data['remanence']['field'].v,
            mobj.data['remanence']['mag'].v,
            **plt_props)
