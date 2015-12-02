__author__ = 'mike'
import RockPy3.Packages.Generic.Features.generic


def backfield_data(ax, mobj, **plt_props):
    ax.plot(mobj.data['data']['field'].v,
            mobj.data['data']['mag'].v,
            **plt_props)
