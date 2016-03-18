__author__ = 'mike'


def plot_x_y(ax, xdata, ydata, **plt_props):
    ax.plot(xdata, ydata, **plt_props)

def plot_errorbars(ax, xdata, ydata, xerr=None, yerr=None, **plt_props):
    ax.errorbar(xdata, ydata, xerr, yerr, **plt_props)


def text_x_y(ax, s, x, y, **plt_props):
    """
    plots text in data coordinates
    :param ax:
    :param text:
    :param x:
    :param y:
    :param plt_props:
    :return:
    """
    if 'transform' in plt_props:
        if plt_props['transform'] == 'ax':
            plt_props['transform'] = ax.transAxes
    ax.text(x, y, s, **plt_props)
