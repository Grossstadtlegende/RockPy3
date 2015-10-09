import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import scipy.interpolate
from scipy import stats

from RockPy.core import measurement
import RockPy

if __name__ == '__main__':
    dfile = '/Users/mike/Dropbox/experimental_data/FeNi/FeNi20H/FeNi_FeNi20-Ha36e02-G01_FORC_VSM#[]_[]_[]#-.001'
    s = RockPy.Sample('test_sample')
    m = s.add_measurement(mtype='forc_old', fpath=dfile, ftype='vsm')
    m.plt_drift_moment_sat()
    m.plt_forc()

class Forc_old(measurement.Measurement):
    def __fitPolySurface(self, data):
        x, y, z = data
        v = np.array([np.ones(len(x)), x, y, x ** 2, y ** 2, x * y])
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        coefficients, residues, rank, singval = np.linalg.lstsq(v.T, z)
        return mean_x, mean_y, -0.5 * coefficients[5], residues[0]

    @property
    def return_fitting_surface(self):  # todo get working
        '''
        function generates a list (fitdata) of lists, where each of the lists contains the value of Ha, Hb and the values of the sub-surface
        :return:
        '''

        x_grid, y_grid = np.meshgrid(self.xi, self.yi)
        fitdata = [[
                       np.array(y_grid[Hac - self.SF:Hac + self.SF + 1, Hbc - self.SF:Hbc + self.SF + 1]).flatten(),
                       # Ha
                       np.array(x_grid[Hac - self.SF:Hac + self.SF + 1, Hbc - self.SF:Hbc + self.SF + 1]).flatten(),
                       # Hb
                       np.array(self.zi[Hac - self.SF:Hac + self.SF + 1, Hbc - self.SF:Hbc + self.SF + 1]).flatten()]
                   # values
                   for Hac in xrange(self.SF, self.len_ha - self.SF)
                   for Hbc in xrange(self.SF, self.len_hb - self.SF)]

        # # if self.zi[Hac - self.SF:Hac + self.SF + 1, Hbc - self.SF:Hbc + self.SF + 1].shape == (2 * self.SF + 1, 2 * self.SF + 1)]
        #
        # fitdata = [i for i in fitdata if np.count_nonzero(np.isnan(i[2])) < 0.5 * len(i[2].flatten())]
        fitdata = [i for i in fitdata if np.count_nonzero(np.isnan(i[2])) < 0.01 * len(i[2].flatten())]
        return np.array(fitdata)

    def format_vsm(self):
        self.raw_data = self.ftype_data.forc()
        self.temperature = [i[:, self.ftype_data.data_header_idx['temperature']] for i in self.raw_data]
        self.field = [i[:, self.ftype_data.data_header_idx['field']] for i in self.raw_data]
        self.moment = [i[:, self.ftype_data.data_header_idx['moment']] for i in self.raw_data]

    def __init__(self, sample_obj, mtype, fpath, ftype, **options):
        log = 'RockPy.MEASUREMENT.forc'
        super(Forc_old, self).__init__(sample_obj, mtype, fpath, ftype, **options)

        self.fitted_forc_data = None
        self.SF = 3

        script = self.ftype_data.measurement_header['SCRIPT']

        self.h_sat = float(script.get('HSat', None))

        if self.temperature is None:
            self.temperature = [np.ones(len(i)) * 295 for i in
                                    self.moment]  # generates temp = 295C data if temp data not in data file

        nforc = int(script['NForc'])
        start = len(self.field) - 2 * nforc

        self.drift = np.array(
            [np.c_[[float(self.field[i]), float(self.moment[i]), float(self.temperature[i])]] for i in
             range(start, len(self.moment)) if
             (i + (start % 2)) % 2 == 0])
        self.drift = self.drift.reshape((self.drift.shape[0], self.drift.shape[1]))
        self.drift_std = np.std(self.drift, axis=0)

        self.forcs = np.array(
            [np.array([self.field[i], self.moment[i]]) for i in range(start, len(self.moment))  # todo temperature
             if (i + (start % 2)) % 2 != 0])

        self.len_ha = len(self.forcs)
        self.len_hb = max([len(i) for i in self.moment])

        self.field_spacing_aux = np.array(
            [np.diff(self.field[i]) for i in range(start, len(self.moment)) if (i + (start % 2)) % 2 != 0][
            1:]).flatten()

        self.field_spacing = np.array([j for i in self.field_spacing_aux for j in i[1:-1]])

        self.field_spacing_first = np.array([i[0] for i in self.field_spacing_aux])
        self.field_spacing_last = np.array([i[-1] for i in self.field_spacing_aux])
        self.field_spacing_std = np.std(self.field_spacing)

        forc_data = []
        hysteresis_df = []
        hysteresis_uf = []

        append = hysteresis_df.append
        for i in self.forcs:
            ha = i[0][0]
            mha = i[1][0]
            append(np.array([ha, mha]))
            for j in range(len(i[0])):
                hb = i[0][j]
                mhb = i[1][j]
                # t = i[2][j]

                aux = np.array([ha, hb, mhb])
                forc_data.append(aux)
                if not j == 0:
                    mirror = np.array([aux[0], 2 * aux[0] - aux[1], 2 * mha - mhb])  # todo incorp temperature
                    forc_data.append(mirror)

        self.forc_data = np.array(forc_data)
        self.hysteresis_df = np.array(hysteresis_df)
        self.hysteresis_uf = self.forcs[-1]

        # todo calculate backfield from data and add coe measurement
        self.backfield = self.calculate_backfield()
        # todo calculate hysteresis from data
        # print self.forcs[-1]
        # self.backfield = np.array(backfield)

        y = self.forc_data[:, 0]  # Ha
        x = self.forc_data[:, 1]  # Hb
        z = self.forc_data[:, 2]

        self.xi = np.linspace(self.min_Hb, self.max_Hb, self.len_hb)
        self.yi = np.linspace(self.min_Ha, self.max_Ha, self.len_ha)
        self.zi = scipy.interpolate.griddata((x, y), z, (self.xi[None,:], self.yi[:,None]), method='linear')

        # self.zi = griddata(x, y, z, self.xi, self.yi, interp='linear')  # grid the data

    def calculate_backfield(self):
        '''
        calculates backfield component from forc data. Returns data with [ha, m(B=0), sm(from regression)]
        M(B=0) is calculated using a linear regression around 0 (+- 3 data points) for each forc branch.
        :return: ndarray
        '''

        aux = []
        n = 0
        for i in self.forcs:
            n += 1
            idx = np.argmin(np.fabs(i[0]))
            ha = min(i[0])
            if ha < 0:
                x = i[0][idx - 3:idx + 3]
                y = i[1][idx - 3:idx + 3]
                if len(x) > 0:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                    mx = x.mean()
                    # my = y.mean()
                    sx2 = sum(((x - mx) ** 2))
                    i = np.sqrt(1. / sx2)

                    # sxy = sum((x-mx)*(y-my))
                    # slope = sxy / sx2
                    # intercept = my - slope * mx
                    # y_new = x * slope + intercept
                    # sse = sum((y-y_new)**2)
                    # std_err = sse / (len(x)-2)
                    sd_intercept = (std_err / i) * np.sqrt(1. / len(x) + mx ** 2 * mx / sx2)
                    aux.append(np.array([ha, intercept, 2 * sd_intercept]))
        out = np.array(aux)
        return out

    def calculate_forc(self, SF=3, **options):
        # self.log.info('CACULATING forc data with smoothing factor << %i >>' % SF)
        self.SF = SF

        # todo multiprocessing
        # pool = multiprocessing.Pool(processes=6)
        # self.fitted_forc_data = np.array(pool.map(self.__fitPolySurface, self.return_fitting_surface))
        # pool.close()
        self.fitted_forc_data = np.array(map(self.__fitPolySurface, self.return_fitting_surface))

    def plt_backfield(self):
        # print self.backfield
        plt.axhline(color='#808080')
        plt.axvline(color='#808080')
        plt.plot(self.backfield[:, 0], self.backfield[:, 1], '.-')
        plt.xlim([min(self.backfield[:, 0]), max(self.backfield[:, 0])])
        plt.title('Backfield: %s' % self.sample_obj.name)
        plt.xlabel('Field [T]')
        plt.ylabel('Moment')
        plt.show()


    @property
    def TranslateHaHbHcHu(self):
        ''' translate from Ha,Hb coordinates into common Hu,Hc coordinats'''
        # Hu = (Ha+Hb) / 2
        # Hc = (Hb-Ha) / 2
        HcHudata = None

        if self.fitted_forc_data is None:
            return

        for fl in self.fitted_forc_data:
            Hu = (fl[0] + fl[1]) / 2
            Hc = (fl[1] - fl[0]) / 2

            newfl = (Hc, Hu, fl[2], fl[3])

            if HcHudata == None:
                HcHudata = np.reshape(newfl, (1, len(newfl)))  # reshape needed to form 2D array
            else:
                HcHudata = np.vstack(( HcHudata, newfl))  # append line to array

        return HcHudata


    @property
    def min_Ha(self):
        return min(self.forc_data[:, 0])

    @property
    def min_Hb(self):
        return min(self.forc_data[:, 1])

    @property
    def max_Ha(self):
        return max(self.forc_data[:, 0])

    @property
    def max_Hb(self):
        return max(self.forc_data[:, 1])

    def plt_forcs(self, n=5):

        for i in range(0, len(self.forcs), n):
            plt.plot(self.forcs[i][0], self.forcs[i][1]
            )
        MX = max([max(i[0]) for i in self.forcs])
        MXy = max([max(i[1]) for i in self.forcs])
        plt.xlim(-MX, MX)
        plt.ylim(-MXy, MXy)
        plt.show()

    def plt_ha_hb_space(self):

        plt.grid()

        # plt.imshow( zi, origin='lower', extent=(minHb, maxHb, minHa, maxHa), cmap='jet')
        plt.contourf(self.xi, self.yi, self.zi, 30, cmap=plt.cm.get_cmap("jet"), zorder=100)
        plt.colorbar()  # draw colorbar
        plt.xlabel('Hb')
        plt.ylabel('Ha')
        plt.title('FORC raw data (magnetic moments)')
        plt.axis('equal')
        plt.show()

    def plt_forc_field_spacing(self):
        """
        plots the saturating field and standard deviation

        """
        plt.title('Statistical distribution of Field Steps')
        plt.hist(self.field_spacing_first, 50, normed=0, facecolor='red', alpha=0.5, log=True, label='first')
        n, bins, patches = plt.hist(self.field_spacing, 50, normed=0, facecolor='green', alpha=0.5, log=True,
                                    label='other')
        plt.hist(self.field_spacing_last, 50, normed=0, facecolor='blue', alpha=0.5, log=True, label='last')
        plt.xlabel('Step size [T]')
        plt.ylabel('Number of steps')
        plt.legend()
        plt.show()


    def plt_drift_field_sat(self):
        """
        plots the saturating field and standard deviation

        """

        plt.plot(range(len(self.drift[:, 0])), self.drift[:, 0] / self.h_sat * 100, '.')
        plt.plot(range(len(self.drift[:, 0])),
                 self.drift[:, 0] / self.h_sat * 100 + self.drift_std[0] / self.h_sat * 100, '-', color='r', alpha=0.4)
        plt.plot(range(len(self.drift[:, 0])),
                 self.drift[:, 0] / self.h_sat * 100 - self.drift_std[0] / self.h_sat * 100, '-', color='r', alpha=0.4)
        plt.fill_between(range(len(self.drift[:, 0])),
                         self.drift[:, 0] / self.h_sat * 100 + self.drift_std[0] / self.h_sat * 100,
                         self.drift[:, 0] / self.h_sat * 100 - self.drift_std[0] / self.h_sat * 100,
                         color='r', alpha=0.2)
        plt.title('Drift of field at $H_{sat}$')
        plt.xlabel('calibration measurement number')
        plt.ylabel('deviation of set $H_{sat}$')
        plt.ticklabel_format(axis='y', offset=None)
        plt.show()

    def plt_drift_moment_sat(self):
        """
        plots the saturation moment and standard deviation

        """
        data = self.drift[:, 1] / max(self.drift[:, 1])
        plt.plot(range(len(data)), data, '.')
        plt.plot(range(len(data)), data + self.drift_std[1] / max(self.drift[:, 1]), '-', color='r', alpha=0.4)
        plt.plot(range(len(data)), data - self.drift_std[1] / max(self.drift[:, 1]), '-', color='r', alpha=0.4)
        plt.fill_between(range(len(data)), data + self.drift_std[1] / max(self.drift[:, 1]),
                         data - self.drift_std[1] / max(self.drift[:, 1]),
                         color='r', alpha=0.2)
        plt.title('Drift of moment at $H_{sat}$')
        plt.xlim(0, len(data))
        plt.xlabel('calibration measurement number')
        plt.ylabel('moment calibration value')
        plt.show()

    def plt_forc(self, SF=2):

        if self.fitted_forc_data is None or SF != self.SF:
            # self.log.warning('NO fitted data found, fitting data with smoothing factor << %i >>' % self.SF)
            self.calculate_forc(SF=SF)

        # plot FORC diagram in Ha, Hb
        # x, y, z = self.fitted_forc_data[:, 1], self.fitted_forc_data[:, 0], self.fitted_forc_data[:, 2]
        #
        # # define grid
        # xi = np.linspace(min(x),max(x), self.len_hb)
        # yi = np.linspace(min(y),max(y), self.len_ha)
        #
        # # grid the data
        # zi = griddata(x, y, z, xi, yi, interp='linear')
        #
        # plt.figure(2)
        # # contour the gridded data
        # plt.contourf(xi, yi, zi, 20, cmap=plt.cm.get_cmap("jet"))
        # #plt.imshow( zi, origin='lower', extent=(minHb, maxHb, minHa, maxHa), cmap='RdBu')
        # plt.colorbar()  # draw colorbar
        # plt.xlabel('Hb')
        # plt.ylabel('Ha')
        # plt.title('FORC processed (SF=%d)' % self.SF)
        # plt.axis('equal')
        from matplotlib.colors import Normalize

        class MidpointNormalize(Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                # I'm ignoring masked values and all kinds of edge cases to make a
                # simple example...
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        xyz = self.TranslateHaHbHcHu
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        # define grid

        xi = np.linspace(0, max(x), self.len_ha * 2)  # Hc space
        yi = np.linspace(min(y), max(y), self.len_hb * 2)  # Hu space
        # grid the data
        grid_x, grid_y = np.mgrid[0:max(x):self.len_ha * 2, min(y): max(y):self.len_hb * 2]
        zi = scipy.interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
        plt.figure(3)
        # contour the gridded data
        norm = MidpointNormalize(midpoint=0)
        plt.contourf(xi, yi, zi, 50,
                     norm=norm,
                     # cmap=plt.cm.get_cmap("Spectral"),
                     cmap=plt.cm.seismic
        )

        # plt.axhline(0, color='black')
        plt.colorbar()  # draw colorbar
        # plot data points.
        plt.xlabel('Hc')
        plt.ylabel('Hu')
        # show data points
        # plt.scatter( x, y)
        plt.title('FORC processed (SF=%d)' % self.SF)
        # plt.axis('equal')
        plt.axis('scaled')

        plt.xlim([sorted(xi)[50], xi.max()])
        plt.ylim([0.7 * sorted(yi)[50], yi.max()])
        # plt.savefig( fname + '.png')

        plt.show()

    def plt_residuals(self):
        xyz = self.TranslateHaHbHcHu
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 3]
        # define grid

        xi = np.linspace(0, max(x), self.len_ha * 2)  # Hc space
        yi = np.linspace(min(y), max(y), self.len_hb * 2)  # Hu space
        # grid the data
        zi = griddata(x, y, z, xi, yi, interp='linear')

        plt.figure(3)
        # contour the gridded data
        plt.contourf(xi, yi, zi, 50, cmap=plt.cm.get_cmap("jet"))
        plt.axhline(0, color='black')
        plt.colorbar()  # draw colorbar
        # plot data points.
        plt.xlabel('Hc')
        plt.ylabel('Hu')
        # show data points
        # plt.scatter( x, y)
        plt.title('FORC processed (SF=%d)' % self.SF)
        # plt.axis('equal')

        plt.xlim([0, 0.1])
        plt.ylim([-0.1, 0.03])
        # plt.savefig( fname + '.png')

        plt.show()

    def plt_hysteresis(self):
        plt.plot(self.hysteresis_df[:, 0], self.hysteresis_df[:, 1])
        plt.plot(self.hysteresis_uf[0], self.hysteresis_uf[1])
        plt.show()