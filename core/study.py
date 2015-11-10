import time

import tabulate

import RockPy3
from RockPy3.core import utils
import RockPy3.core.sample
import RockPy3.core.file_operations
import os
from multiprocessing import Pool
from copy import deepcopy

class Study(object):
    """
    comprises data of a whole study
    i.e. container for samplegroups
    """

    def __init__(self, name=None):
        """
        general Container for Samplegroups and Samples.

        Parameters
        ----------
            name: str
                default: 'study'
                name of the Study
        """
        if not name:
            name = time.strftime("%Y%m%d:%H%M")
        # self.log = log  # logging.getLogger('RockPy3.' + type(self).__name__)
        RockPy3.logger.info('CREATING study << {} >>'.format(name))
        self.name = name
        self._samples = dict()  # {'sname':'sobj'}

        self._series = {'none': []}  # {series_obj: measurement}

    def __repr__(self):
        if self == RockPy3.Study:
            return '<< RockPy3.MasterStudy >>'.format(self.name)
        else:
            return '<< RockPy3.Study.{} >>'.format(self.name)

    def __getitem__(self, item):
        if item in self._samples:
            return self._samples[item]
        elif item in self.groupnames:
            return self.get_sample(gname=item)
        else:
            try:
                return sorted(self.samplelist)[item]
            except IndexError:
                raise IndexError('index too high {}>{}'.format(item, len(self._samples)))

    @property
    def samplelist(self):
        """
        Returns a list of all samples

        Returns
        -------
            list of all samples
        """

        return sorted([v for k, v in self._samples.items()])

    @property
    def samplenames(self):
        """
        Returns a list of all samplenames

        Returns
        -------
            list of all samplenames
        """
        return [k for k, v in self._samples.items()]

    @property
    def ngroups(self):
        return len(self.groupnames)

    @property
    def groupnames(self):
        return sorted(list(set(i for j in self.samplelist for i in j._samplegroups)))

    @property
    def n_samples(self):
        return len(self._samples)

    ####################################################################################################################
    ''' add functions '''

    def add_sample(self,
                   name=None,
                   comment='',
                   mass=None, mass_unit='kg',
                   height=None, diameter=None,
                   x_len=None, y_len=None, z_len=None,  # for cubic samples
                   length_unit='mm',
                   sample_shape='cylinder',
                   coord=None,
                   samplegroup=None,
                   sobj=None,
                   ):

        if name in self.samplenames:
            RockPy3.logger.warning('CANT create << %s >> already in Study. Please use unique sample names. '
                                   'Returning sample' % name)
            return self._samples[name]

        if not sobj:
            sobj = RockPy3.core.sample.Sample(
                name=name,
                comment=comment,
                mass=mass, mass_unit=mass_unit,
                height=height, diameter=diameter,
                x_len=x_len, y_len=y_len, z_len=z_len,  # for cubic samples
                length_unit=length_unit,
                sample_shape=sample_shape,
                samplegroup=samplegroup,
                coord=coord,
            )

        self._samples.setdefault(sobj.name, sobj)
        return sobj

    def add_samplegroup(self,
                        gname=None,
                        sname=None,
                        mtype=None,
                        series=None,
                        stype=None, sval=None, sval_range=None,
                        mean=False,
                        invert=False,
                        ):
        """
        adds selected samples to a samplegroup

        Parameter
        ---------
            name: str
            default: None
            if None, name is 'SampleGroup #samplegroups'

        Returns
        -------
            list
                list of samples in samplegroup
        """
        samples = self.get_sample(
            sname=sname,
            mtype=mtype,
            series=series,
            stype=stype, sval=sval, sval_range=sval_range,
            mean=mean,
            invert=invert,
        )
        if not gname:
            gname = 'SG%02i' % self.ngroups

        for s in samples:
            s.add_to_samplegroup(gname=gname)

        return samples

    def add_mean_sample(self):
        pass

    ####################################################################################################################
    ''' remove functions '''

    def remove_samplegroup(self, gname=None):
        """
        removes all samples from a samplegroup and therefore the samplegroup itself
        :param gname: samplegroup name
        :return:
        """
        samples = self.get_sample(gname=gname)
        for s in samples:
            s.remove_from_samplegroup(gname=gname)

    ####################################################################################################################
    ''' get functions '''

    def get_sample(self,
                   gname=None,
                   sname=None,
                   mtype=None,
                   series=None,
                   stype=None, sval=None, sval_range=None,
                   mean=False,
                   invert=False,
                   ):

        slist = self.samplelist

        if not any(i for i in [gname, sname, mtype, series, stype, sval, sval_range, mean, invert]):
            return slist

        # samplegroup filtering
        if gname:
            gname = utils.to_list(gname)
            slist = [s for s in slist if any(sg in gname for sg in s._samplegroups)]

        # sample filtering
        if sname:
            sname = utils.to_list(sname)
            slist = [s for s in slist if s.name in sname]

        slist = [s for s in slist if s.get_measurement(mtype=mtype,
                                                       stype=stype, sval=sval,
                                                       sval_range=sval_range, series=series)]
        # # mtype filtering
        # if mtype:
        #     mtype = utils.to_list(mtype)
        #     slist = [s for s in slist if any(mt in mtype for mt in s.mtypes)]
        #
        # # stype filtering
        # if stype:
        #     stype = utils.to_list(stype)
        #     slist = [s for s in slist if any(mt in stype for mt in s.stypes)]
        #
        # # sval filtering
        # if stype:
        #     stype = utils.to_list(stype)
        #     slist = [s for s in slist if any(mt in stype for mt in s.stypes)]


        return slist

    def get_samplegroup(self, gname=None):
        """
        wrapper for simply getting all samples of one samplegroup
        :param gname: str
            name of the samplegroup
        :return: list
            list of samples in that group
        """
        return self.get_sample(gname=gname)

    def get_measurement(self,
                        gname=None,
                        sname=None,
                        mtype=None,
                        series=None,
                        stype=None, sval=None, sval_range=None,
                        mean=False,
                        invert=False,
                        id=None,
                        ):

        if id:
            mlist = [m for s in self.samplelist for m in s.get_measurement(id=id, invert=invert)]

        else:
            samples = self.get_sample(gname=gname, sname=sname, mtype=mtype, series=series,
                                      stype=stype, sval=sval, sval_range=sval_range, mean=mean, invert=invert)

            mlist = [m for s in samples for m in s.get_measurement(mtype=mtype, series=series,
                                                                   stype=stype, sval=sval, sval_range=sval_range,
                                                                   mean=mean,
                                                                   invert=invert)]
        return mlist

    def import_folder(self, folder):
        files = [os.path.join(folder, i) for i in os.listdir(folder)
                 if not i.startswith('#')
                 if not i.startswith(".")
                 if not os.path.isdir(os.path.join(folder, i))
                 ]

        sample_groups = set(os.path.basename(f).split('_')[0] for f in files)
        RockPy3.logger.debug(
            'TRYING to import {} files for these samplegroups {}'.format(len(files), sorted(list(sample_groups))))

        start = time.clock()

        # with Pool(5) as p:
        #     measurements = p.map(self.import_file, files)
        # print(measurements)
        # print(measurements[0].sobj)
        measurements = [self.import_file(file) for file in files]
        end = time.clock()
        measurements = [m for m in measurements if m]
        RockPy3.logger.debug(
            'IMPORT generated {} measurements: finished in {:<3}s'.format(len(measurements), end - start))
        return measurements

    def import_file(self, fpath):
        info = RockPy3.get_info_from_fname(fpath)
        sample_info = deepcopy(info)
        if not info['mtype'] in RockPy3.implemented_measurements:
            return
        # remove unnecessary info
        for arg in ['series', 'idx', 'mtype', 'ftype', 'fpath']:
            sample_info.pop(arg, None)
        name = sample_info.pop('sample_name', None)
        if not name in self._samples:
            s = self.add_sample(name=name, **sample_info)
        else:
            s = self._samples[name]
        m = s.add_measurement(**info)
        return m

    def info(self, tablefmt='simple', parameters=True):
        formats = ['plain', 'simple', 'grid', 'fancy_grid', 'pipe', 'orgtbl', 'rst', 'mediawiki', 'html', 'latex',
                   'latex_booktabs']

        if not tablefmt in formats:
            RockPy3.logger.info('NO SUCH FORMAT')
            tablefmt = 'simple'

        header = ['Sample Name', 'Sample Group', 'Measurements', 'series', 'Initial State']
        table = []

        for s in sorted(self.samplelist):
            # get all mtypes of the sample
            mtypes = list(s.mdict['mtype'].keys())
            # get all stypes of the sample
            stypes = ', '.join(list(s.mdict['stype'].keys()))

            # count how many measurements for each mtype
            measurements = ', '.join(['%ix %s' % (len(s.mdict['mtype'][i]), i) for i in mtypes])

            # does the measurement have an initial state
            i_state = [True if any(m.has_initial_state for m in s.measurements) else False][0]

            samplegroups = s._samplegroups
            if not samplegroups:
                samplegroups = 'none'

            # header line for the sample
            line0 = [s.name, samplegroups, measurements, stypes, i_state]
            table.append(line0)
            table.append([''.join(['=' for i in range(15)]) for j in line0])

            # add a line for each measurement
            for m in s.measurements:
                # if not isinstance(m, RockPy3.Packages.Generic.Measurements.parameters.Parameter) or parameters:
                    if m.has_initial_state:
                        initial = '{} [{}]'.format(m.initial_state.mtype, str(m.initial_state.idx))
                    else:
                        initial = ''
                    line = ['', s.name, '{} [{}]'.format(m.mtype, str(m.idx)),
                            ', '.join(['{} [{}]'.format(series[0], series[1]) for series in m.stype_sval_tuples]),
                            initial]
                    table.append(line)
            table.append([''.join(['-' for i in range(15)]) for j in line0])

        return tabulate.tabulate(table, headers=header, tablefmt=tablefmt)

    def calc_all(self, **parameter):
        for s in self.samplelist:
            s.calc_all(**parameter)

    ####################################################################################################################
    ''' Data Operations '''

    def save(self, file_name=None, folder=None):
        if not file_name:
            import time
            file_name = '_'.join([time.strftime("%Y%m%d"), 'RockPy', self.name,
                                  '[{}]SG_[{}]S'.format(len(self._samplegroups), len(self.samples)), '.rpy'])
        if not folder:
            folder = RockPy3.core.file_operations.default_folder
        RockPy3.logger.info('SAVING RockPy data to {}'.format(os.path.join(folder, file_name)))
        RockPy3.save(self, folder=folder, file_name=file_name)

    def load(self, file_name=None, folder=None):
        return RockPy3.load(folder=folder, file_name=file_name)
