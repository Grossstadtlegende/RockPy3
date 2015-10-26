__author__ = 'mike'


def plot_x_y(ax, xdata, ydata, **plt_props):
    ax.plot(xdata, ydata, **plt_props)


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
    print(plt_props)
    ax.text(x, y, s, **plt_props)
