__author__ = 'Mike'

import RockPy3
import numpy as np

def thermocurve_data(ax, mobj, dtype='mag', **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    for dtype in mobj.data:
        if 'warming' in dtype:
            plt_props.update({'color': '#DC3522'})
        if 'cooling' in dtype:
            plt_props.update({'color': '#00305A'})
        ax.plot(mobj.data[dtype]['temp'].v,
                mobj.data[dtype]['mag'].v,
                **plt_props)


def thermocurve_data_colored(ax, mobj, dtype='mag', **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    for dtype in mobj.data:
        ax.plot(mobj.data[dtype]['temp'].v,
                mobj.data[dtype]['mag'].v,
                **plt_props)


def thermocurve_derivative(ax, mobj, twinx=True, **plt_props):
    """
    Plots the down_field branch of a hysteresis
    """
    if twinx:
        ax = ax.twinx()
        ax.set_xlabel(plt_props.get('xlabel'))
        ax.set_ylabel(plt_props.get('ylabel'))

    for dtype in mobj.data:
        x = mobj.data[dtype]['temp'].v
        dx = np.gradient(x)
        dy = np.gradient(mobj.data[dtype]['mag'].v, dx)
        if 'warming' in dtype:
            plt_props.update({'color': '#DC3522'})
        if 'cooling' in dtype:
            plt_props.update({'color': '#00305A'})
        if not 'ls' in plt_props:
            plt_props.pop('linestyle')
            plt_props.update({'ls': '--'})
        ax.plot(x, dy, **plt_props)

if __name__ == '__main__':
    import RockPy3
    import pandas
    pandas.DataFrame
    S = RockPy3.Study
    S.import_folder('/Users/Mike/Dropbox/experimental_data/pyrrhotite/rmp_||a')
    S.import_folder('/Users/Mike/Dropbox/experimental_data/pyrrhotite/rmp_||b')
    S.import_folder('/Users/Mike/Dropbox/experimental_data/pyrrhotite/rmp_||c')
    print(S[0]['axis' == 2])