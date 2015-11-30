__author__ = 'mike'
import RockPy3.Packages.Generic.Features.generic
import numpy as np


def day_data(ax, mobj, **plt_props):
    print(mobj)
    if not len(mobj) == 2:
        return
    ms = mobj[0].result_ms()
    mrs = mobj[0].result_mrs()
    bc = mobj[0].result_bc()
    bcr = mobj[1].result_bcr()
    ax.plot(bcr[0] / bc[0], mrs[0] / ms[0], **plt_props)


def sd_md_mixline_1(ax, **plt_opt):
    color = plt_opt.pop('color', 'k')
    marker = plt_opt.pop('marker', '.')
    ls = plt_opt.pop('ls', '--')
    zorder = plt_opt.pop('zorder', 0)
    fontsize = plt_opt.pop('fontsize', 10)

    text_out = []  # for multiple texts that get plotted

    data = np.array(
            [[1.259, 0.500], [1.296, 0.448], [1.337, 0.404], [1.404, 0.353], [1.473, 0.305], [1.569, 0.259],
             [1.704, 0.211], [1.913, 0.163], [2.275, 0.114], [2.556, 0.090], [3.012, 0.067], [3.767, 0.043],
             [4.601, 0.029], [5.366, 0.019]])

    text = {'texts': ['0%', '20%', '40%', '60%', '80%', '90%', '95%', '98%', '100%'],
            'positions': data[[0, 2, 4, 6, 8, 10, 11, 12, 13]]}

    line_out = ax.plot(data[:, 0], data[:, 1], color=color, marker=marker, ls=ls, zorder=zorder, **plt_opt)

    for i, t in enumerate(text['texts']):
        text_out.append(ax.text(text['positions'][i][0], text['positions'][i][1], t,
                                verticalalignment='top', horizontalalignment='right',
                                color=color, fontsize=fontsize))

    return line_out, text_out


def sd_md_mixline_2(ax, **plt_opt):
    color = plt_opt.pop('color', 'k')
    marker = plt_opt.pop('marker', '.')
    ls = plt_opt.pop('ls', '--')
    zorder = plt_opt.pop('zorder', 0)
    fontsize = plt_opt.pop('fontsize', 10)

    text_out = []  # for multiple texts that get plotted

    data = np.array(
            [[1.431, 0.378], [1.508, 0.341], [1.601, 0.306], [1.702, 0.270], [1.810, 0.234], [1.973, 0.198],
             [2.186, 0.161], [2.494, 0.125], [2.928, 0.090], [3.214, 0.072], [3.646, 0.053], [4.151, 0.036],
             [5.025, 0.018]])

    text = {'texts': ['0%', '20%', '40%', '60%', '80%', '90%', '95%', '100%'],
            'positions': data[[0, 2, 4, 6, 8, 10, 11, 12]]}

    line_out = ax.plot(data[:, 0], data[:, 1], color=color, marker=marker, ls=ls, zorder=zorder, **plt_opt)

    for i, t in enumerate(text['texts']):
        text_out.append(ax.text(text['positions'][i][0] + 0.1, text['positions'][i][1] + 0.005, t,
                                verticalalignment='bottom', horizontalalignment='left',
                                color=color, fontsize=fontsize))

    return line_out, text_out


def sp_envelope(ax, **plt_opt):
    color = plt_opt.pop('color', 'k')
    marker = plt_opt.pop('marker', '.')
    ls = plt_opt.pop('ls', '--')
    zorder = plt_opt.pop('zorder', 0)
    fontsize = plt_opt.pop('fontsize', 10)

    text_out = []  # for multiple texts that get plotted

    data = np.array(
            [[1.255, 0.498], [1.414, 0.473], [1.622, 0.450], [1.945, 0.426], [2.512, 0.402], [3.764, 0.376],
             [7.411, 0.352], [8.874, 0.350], [17.269, 0.343], [34.897, 0.338], [57.714, 0.336]])

    line_out = ax.plot(data[:, 0], data[:, 1], color=color, marker=marker, ls=ls, zorder=zorder, **plt_opt)

    text_out.append(ax.text(3.8, 0.38, 'SP saturation envelope',
                            verticalalignment='bottom', horizontalalignment='left',
                            color=color, fontsize=fontsize))

    return line_out, text_out


def sd_sp_10nm(ax, **plt_opt):
    color = plt_opt.pop('color', 'k')
    marker = plt_opt.pop('marker', '.')
    ls = plt_opt.pop('ls', '--')
    zorder = plt_opt.pop('zorder', 0)
    fontsize = plt_opt.pop('fontsize', 10)

    text_out = []  # for multiple texts that get plotted

    data = np.array(
            [[1.942, 0.426], [2.193, 0.409], [2.726, 0.375], [3.135, 0.350], [3.619, 0.323], [4.175, 0.299],
             [4.879, 0.273], [5.617, 0.250], [6.605, 0.224], [7.859, 0.200], [9.426, 0.174], [11.432, 0.150],
             [14.353, 0.125], [18.754, 0.100]])

    text = {'texts': ['20%', '40%', '60%', '80%'],
            'positions': data[[1, 5, 9, 13]]}

    line_out = ax.plot(data[:, 0], data[:, 1], color=color, marker=marker, ls=ls, zorder=zorder, **plt_opt)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for i, t in enumerate(text['texts']):
        x, y = text['positions'][i][0] + 0.1, text['positions'][i][1] + 0.005
        if xmin < x * 1.1 < xmax and ymin < y * 0.9 < ymax:
            ax.text(x, y, t,
                    verticalalignment='bottom', horizontalalignment='left',
                    color=color, fontsize=fontsize)

    ax.text(xmax * 0.85, text['positions'][-1][1] - 0.005, '10 nm',
            verticalalignment='bottom', horizontalalignment='left',
            color=color, fontsize=fontsize)

    return line_out, text_out


def sd_sp_15nm(ax, **plt_opt):
    color = plt_opt.pop('color', 'k')
    marker = plt_opt.pop('marker', '.')
    ls = plt_opt.pop('ls', '--')
    zorder = plt_opt.pop('zorder', 0)
    fontsize = plt_opt.pop('fontsize', 10)

    text_out = []  # for multiple texts that get plotted

    data = np.array(
            [[1.942, 0.426], [2.193, 0.409], [2.359, 0.401], [2.726, 0.375], [3.135, 0.350], [3.619, 0.323],
             [4.175, 0.299],
             [4.879, 0.273], [5.617, 0.250], [6.605, 0.224], [7.859, 0.200], [9.426, 0.174], [11.432, 0.150],
             [14.353, 0.125], [18.754, 0.100]])

    text = {'texts': ['0%', '20%', '40%', '60%', '80%', '90%', '95%', '100%'],
            'positions': data[[0, 2, 4, 6, 8, 10, 11, 12]]}

    line_out = ax.plot(data[:, 0], data[:, 1], color=color, marker=marker, ls=ls, zorder=zorder, **plt_opt)

    for i, t in enumerate(text['texts']):
        text_out.append(ax.text(text['positions'][i][0] + 0.1, text['positions'][i][1] + 0.005, t,
                                verticalalignment='bottom', horizontalalignment='left',
                                color=color, fontsize=fontsize))
    text_out.append(ax.text(text['positions'][-1][0], text['positions'][-1][1] - 0.005, '10 nm',
                            verticalalignment='bottom', horizontalalignment='left',
                            color=color, fontsize=fontsize))

    return line_out, text_out


def langevine_mixline(ax, **plt_opt):
    color = plt_opt.pop('color', 'k')
    marker = plt_opt.pop('marker', '.')
    ls = plt_opt.pop('ls', '--')
    zorder = plt_opt.pop('zorder', 0)
    fontsize = plt_opt.pop('fontsize', 10)

    text_out = []  # for multiple texts that get plotted

    data = np.array(
            [[1.259, 0.498], [1.412, 0.472], [1.619, 0.444], [1.941, 0.407], [2.351, 0.371], [2.715, 0.348],
             [3.151, 0.324], [3.628, 0.302], [4.172, 0.281], [4.862, 0.261], [5.629, 0.239], [6.610, 0.217],
             [7.823, 0.194], [9.424, 0.171], [11.432, 0.150], [14.353, 0.125], [18.754, 0.100]])

    line_out = ax.plot(data[:, 0], data[:, 1], color=color, marker=marker, ls=ls, zorder=zorder, **plt_opt)

    text_out = ax.annotate('Langevin\ncalculation', xy=np.mean(data[[6, 7]], axis=0),
                           xytext=np.mean(data[[6, 7]], axis=0) * [0.8, 0.7],
                           # arrowprops=dict(facecolor='black', shrink=0.05, width=0.5),
                           arrowprops=dict(arrowstyle="-"),
                           )

    return line_out, text_out


def sd_sp_93nm(ax, **plt_opt):
    color = plt_opt.pop('color', 'k')
    marker = plt_opt.pop('marker', '.')
    ls = plt_opt.pop('ls', '--')
    zorder = plt_opt.pop('zorder', 0)
    fontsize = plt_opt.pop('fontsize', 10)

    text_out = []  # for multiple texts that get plotted

    data = np.array(
            [[1.259, 0.498], [1.412, 0.472], [1.619, 0.444], [1.941, 0.407], [2.351, 0.371], [2.715, 0.348],
             [3.151, 0.324], [3.628, 0.302], [4.172, 0.281], [4.862, 0.261], [5.629, 0.239], [6.610, 0.217],
             [7.823, 0.194], [9.424, 0.171], [11.432, 0.150], [14.353, 0.125], [18.754, 0.100]])

    line_out = ax.plot(data[:, 0], data[:, 1], color=color, marker=marker, ls=ls, zorder=zorder, **plt_opt)

    text_out = ax.annotate('Langevin\ncalculation', xy=np.mean(data[[6, 7]], axis=0),
                           xytext=np.mean(data[[6, 7]], axis=0) * [0.8, 0.7],
                           # arrowprops=dict(facecolor='black', shrink=0.05, width=0.5),
                           arrowprops=dict(arrowstyle="-"),
                           )

    return line_out, text_out


def day_grid(ax, **options):
    ax.hlines(0.5, 0, 10)
    ax.hlines(0.1, 0, 10)
    ax.vlines(5, 0, 0.6)
    ax.vlines(1.5, 0, 0.6)
    ax.text(0.6, 0.55, 'SD', fontdict={'size': 14, 'color': '#909090'})
    ax.text(2.6, 0.3, 'PSD', fontdict={'size': 14, 'color': '#909090'})
    ax.text(5.9, 0.05, 'MD', fontdict={'size': 14, 'color': '#909090'})
