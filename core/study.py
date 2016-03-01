import time
import tabulate
import xml.etree.ElementTree as etree
from copy import deepcopy
import os
from os.path import join
from multiprocessing import Pool
import logging
import RockPy3
from RockPy3.core import utils
import RockPy3.core.sample
import RockPy3.core.file_operations
import numpy as np

log = logging.getLogger(__name__)


class Study(object):
    """
    comprises data of a whole study
    i.e. container for samplegroups
    """

    # XML tags
    ROCKPY = 'rockpy'
    STUDY = 'study'
    SAMPLES = 'samples'
    NAME = 'nane'

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
        if item in self.samplenames:
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
        return sorted([k for k, v in self._samples.items()])

    @property
    def ngroups(self):
        return len(self.groupnames)

    @property
    def groupnames(self):
        return sorted(set(i for j in self.samplelist for i in j._samplegroups))

    @property
    def samplegroups(self):
        return self.groupnames

    @property
    def n_samples(self):
        return len(self._samples)

    @property
    def stypes(self):
        stypes = set(stype for s in self.samplelist for stype in s.stypes)
        return stypes

    @property
    def stype_svals(self):
        """
        searches through all samples and creates a dictionary with all stype:[svals] of that type
        :return:
        """
        stype_sval_dict = {stype: set() for stype in self.stypes}
        for stype in stype_sval_dict:
            for s in self.samplelist:
                if stype in s.mdict['stype_sval']:
                    stype_sval_dict[stype].update(s.mdict['stype_sval'][stype].keys())
            else:
                stype_sval_dict[stype] = sorted(stype_sval_dict[stype])
        return stype_sval_dict

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
                   **options
                   ):

        if name in self.samplenames:
            RockPy3.logger.warning('CANT create << %s >> already in Study. Please use unique sample names. '
                                   'Returning sample' % name)
            return self._samples[name]

        if not sobj:
            sobj = RockPy3.core.sample.Sample(
                    name=str(name),
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
                        slist=None,
                        ):
        """
        adds selected samples to a samplegroup

        Parameter
        ---------
            name: str
            default: None
            if None, name is 'SampleGroup #samplegroups'
            slist: list
                list of samples to be added to the sample_group

        Returns
        -------
            list
                list of samples in samplegroup
        """
        if not slist:
            samples = self.get_sample(
                    sname=sname,
                    mtype=mtype,
                    series=series,
                    stype=stype, sval=sval, sval_range=sval_range,
                    mean=mean,
                    invert=invert,
            )
        else:
            samples = slist
        if not gname:
            gname = 'SG%02i' % self.ngroups

        for s in samples:
            s.add_to_samplegroup(gname=gname)

        return samples

    def remove_samplegroup(self,
                           gname=None,
                           sname=None,
                           mtype=None,
                           series=None,
                           stype=None, sval=None, sval_range=None,
                           mean=False,
                           invert=False,
                           slist=None,
                           ):
        """
        removes selected samples from a samplegroup

        Parameter
        ---------
            gname: str
                the name of the samplegroup that is supposed to be removed
            slist: list
                list of samples to be removed the sample_group

        Returns
        -------
            list
                list of samples in samplegroup
        """
        if not gname:
            RockPy3.logger.error('NO sample group specified')
            return

        if not slist:
            samples = self.get_sample(
                    sname=sname,
                    mtype=mtype,
                    series=series,
                    stype=stype, sval=sval, sval_range=sval_range,
                    mean=mean,
                    invert=invert,
            )
        else:
            samples = slist

        [s.remove_from_samplegroup(gname) for s in samples]

    def add_mean_sample(self,
                        gname=None,
                        sname=None,
                        mtype=None,
                        series=None,
                        stype=None, sval=None, sval_range=None,
                        mean=False,
                        invert=False,
                        interpolate=True, substfunc='mean', mean_of_mean=False,
                        reference=None, ref_dtype='mag', norm_dtypes='all', vval=None, norm_method='max',
                        normalize_variable=False, dont_normalize=None,
                        ignore_series=False
                        ):
        """
        creates a mean sample from the input search
        :param gname:
        :param sname:
        :param mtype:
        :param series:
        :param stype:
        :param sval:
        :param sval_range:
        :param mean:
        :param invert:
        :param interpolate:
        :param substfunc:
        :param mean_of_mean:
        :param reference:
        :param ref_dtype:
        :param norm_dtypes:
        :param vval:
        :param norm_method:
        :param normalize_variable:
        :param dont_normalize:
        :param ignore_series:
        :return:
        """
        # get specified measurements
        slist = self.get_sample(
                gname=gname, sname=sname, mtype=mtype,
                series=series, stype=stype, sval=sval, sval_range=sval_range,
                mean=mean, invert=invert)

        samples = '_'.join(sobj.name for sobj in slist)

        # create empty mean_sample
        mean_sample = RockPy3.MeanSample(name='mean[{}]'.format(samples))

        # get all measurements from all samples
        for sample in slist:
            # do not use mean samples otherwise strange effects
            if not isinstance(sample, RockPy3.MeanSample):
                # if a sample has more than one measurement you can chose to either mean them first
                # or add them to the measurements
                if mean_of_mean:
                    mean_measurements = sample.add_mean_measurements(ignore_series=ignore_series,
                                                                     interpolate=interpolate, substfunc=substfunc,
                                                                     reference=reference, ref_dtype=ref_dtype,
                                                                     norm_dtypes=norm_dtypes,
                                                                     vval=vval,
                                                                     norm_method=norm_method,
                                                                     normalize_variable=normalize_variable,
                                                                     dont_normalize=dont_normalize)
                    mean_sample.measurements.extend(mean_measurements)
                else:
                    mean_sample.measurements.extend(sample.measurements)

        # mdict needs to be populated
        mean_sample._populate_mdict()
        # the measurements are now used as if they belonged to that sample
        mean_sample.add_mean_measurements(ignore_series=ignore_series,
                                          interpolate=interpolate, substfunc=substfunc,
                                          reference=reference, ref_dtype=ref_dtype, norm_dtypes=norm_dtypes,
                                          vval=vval,
                                          norm_method=norm_method,
                                          normalize_variable=normalize_variable, dont_normalize=dont_normalize)

        mean_sample.base_measurements = mean_sample.measurements
        mean_sample.measurements = []
        # mdict needs to be populated
        mean_sample._mdict = mean_sample._create_mdict()
        mean_sample._populate_mdict(mdict_type='mean_mdict')
        # mean_sample.mean_measurements = []
        self.add_sample(sobj=mean_sample)
        return mean_sample

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

    def remove_sample(self,
                      gname=None,
                      sname=None,
                      mtype=None,
                      series=None,
                      stype=None, sval=None, sval_range=None,
                      mean=False,
                      invert=False,
                      ):

        samples = self.get_sample(gname=gname, sname=sname, mtype=mtype, series=series,
                                  stype=stype, sval=sval, sval_range=sval_range, mean=mean, invert=invert)

        for s in samples:
            self._samples.pop(s.name, None)

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

        if any(i for i in [mtype, series, stype, sval, sval_range, mean, invert]):
            slist = [s for s in slist if s.get_measurement(mtype=mtype,
                                                           stype=stype, sval=sval, sval_range=sval_range,
                                                           series=series,
                                                           mean=mean,
                                                           invert=invert)]
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
                        mean=False, groupmean=False,
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
            if groupmean:
                mlist = filter(lambda x: isinstance(x.sobj, RockPy3.MeanSample), mlist)

        return sorted(set(mlist))

    def import_folder(self, folder, gname=None):  # todo specify samples, mtypes and series for selective import of folder
        files = [os.path.join(folder, i) for i in os.listdir(folder)
                 if not i.startswith('#')
                 if not i.startswith(".")
                 if not os.path.isdir(os.path.join(folder, i))
                 ]

        sample_groups = set(os.path.basename(f).split('_')[0] for f in files)
        RockPy3.logger.debug(
                'TRYING to import {} files for these samplegroups {}'.format(len(files), sorted(list(sample_groups))))

        start = time.clock()

        measurements = [self.import_file(file) for file in sorted(files)]
        # measurements = map(self.import_file, files)
        # measurements = [m for m in measurements if m]

        end = time.clock()

        if gname:
            samples = set(m.sobj for m in measurements)
            self.add_samplegroup(gname=gname, slist=samples)

        RockPy3.logger.info(
                'IMPORT generated {} measurements: finished in {:<3}s'.format(len(measurements), end - start))
        return measurements

    def import_file(self, fpath):
        """
        Import function for a single file.
        :param fpath:
        :return:
            if file not readable it returns None
        """
        try:
            info = RockPy3.get_info_from_fname(fpath)

            if not info['mtype'] in RockPy3.implemented_measurements:
                return

            if not info['sample_name'] in self._samples:
                s = self.add_sample(name=info['sample_name'], mass=info['mass'], mass_unit=info['mass_unit'])
            else:
                s = self._samples[info['sample_name']]
            m = s.add_measurement(fpath=info['fpath'], series=info['series'])
            return m
        except ValueError:
            return

    def info(self, sample_info=True, tablefmt='simple', parameters=True):
        formats = ['plain', 'simple', 'grid', 'fancy_grid', 'pipe', 'orgtbl', 'rst', 'mediawiki', 'html', 'latex',
                   'latex_booktabs']

        if not tablefmt in formats:
            RockPy3.logger.info('NO SUCH FORMAT')
            tablefmt = 'simple'

        table = [['{} samples'.format(len(self.samplelist)), ','.join(self.samplenames), '           ']]
        table.append([''.join(['--' for i in str(j)]) for j in table[0]])

        mtypes = set(mt for s in self.samplelist for mt in s.mtypes)
        for mtype in mtypes:
            measurements = self.get_measurement(mtype=mtype)
            stypes = sorted(list(set([s.stype for m in measurements for s in m.series])))
            table.append(['{} measurements'.format(len(measurements)), mtype])
            table.append(['', '{} series:'.format(len(stypes))])
            for stype in stypes:
                measurements = self.get_measurement(mtype=mtype, stype=stype)
                values = [s.value for m in measurements for s in m.series if s.stype == stype]
                svals = sorted(list(set(values)))
                svals = ['{}[{}]'.format(sval, values.count(sval)) for sval in svals]
                table.append(['', '{}: {}'.format(stype, ', '.join(svals))])

        table.append([''.join(['--' for i in str(j)]) for j in table[0]])

        print(tabulate.tabulate(table, headers=[self.name, '', '', ''], tablefmt=tablefmt))

        if not sample_info:
            return

        print()
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
            for m in s.measurements + s.mean_measurements:
                # if not isinstance(m, RockPy3.Packages.Generic.Measurements.parameters.Parameter) or parameters:
                if m.has_initial_state:
                    initial = '{} [{}]'.format(m.initial_state.mtype, str(m.initial_state.idx))
                else:
                    initial = ''
                if m.is_mean:
                    if isinstance(m.sobj, RockPy3.MeanSample):
                        samples = set(base.sobj.name for base in m.base_measurements)
                        mean = ''.join(['S' + sample + '{}'.format(
                                [base.idx for base in m.base_measurements if base.sobj.name == sample]) for sample in
                                        samples])
                    else:
                        mean = 'mean{}'.format([base.idx for base in m.base_measurements])
                else:
                    mean = ''

                line = ['', s.name, '{}{} [{}]'.format(mean, m.mtype, str(m.idx)),
                        ', '.join(['{} [{}]'.format(series[0], series[1]) for series in m.stype_sval_tuples]),
                        initial]
                table.append(line)
            table.append([''.join(['-' for i in range(15)]) for j in line0])

        print(tabulate.tabulate(table, headers=header, tablefmt=tablefmt))

    def calc_all(self, **parameter):
        for s in self.samplelist:
            s.calc_all(**parameter)

    ####################################################################################################################
    ''' label operations '''

    def label_add_sname(self):
        for s in self.samplelist:
            s.label_add_sname()

    def label_add_series(self, stypes=None, add_stype=True, add_sval=True, add_unit=True,
                         gname=None,
                         sname=None,
                         mtype=None,
                         series=None,
                         stype=None, sval=None, sval_range=None,
                         mean=False, groupmean=False,
                         invert=False,
                         id=None,
                         ):
        for m in self.get_measurement(gname=gname, sname=sname, mtype=mtype, series=series,
                                      stype=stype, sval=sval, sval_range=sval_range,
                                      mean=mean, groupmean=groupmean,
                                      invert=invert, id=id):
            m.label_add_stype(stypes=stypes, add_stype=add_stype, add_sval=add_sval, add_unit=add_unit)

    def color_from_sample(self):
        for i, s in enumerate(self.samplelist):
            color = RockPy3.colorscheme[i]
            s.set_plt_prop('color', color)

    def set_color(self, color, stypes=None, add_stype=True, add_sval=True, add_unit=True,
                  gname=None,
                  sname=None,
                  mtype=None,
                  series=None,
                  stype=None, sval=None, sval_range=None,
                  mean=False, groupmean=False,
                  invert=False,
                  id=None,
                  ):

        for m in self.get_measurement(gname=gname, sname=sname, mtype=mtype, series=series,
                                      stype=stype, sval=sval, sval_range=sval_range,
                                      mean=mean, groupmean=groupmean,
                                      invert=invert, id=id):
            m.set_plt_prop('color', color)

    def color_from_series(self, stype):
        """
        assigns a color to each measurement, according to the value of their series
        :param stype:
        :return:
        """

        # get svals
        svals = self.stype_svals[stype]
        # create heatmap
        color_map = self.create_heat_color_map(svals)

        # set the color for each sval
        for i, sv in enumerate(svals):
            self.set_color(stype=stype, sval=sv, color=color_map[i])

    @staticmethod
    def create_heat_color_map(value_list, reverse=False):
        """
        takes a list of values and creates a list of colors from blue to red (or reversed if reverse = True)

        :param value_list:
        :param reverse:
        :return:
        """
        red = np.linspace(0, 255, len(value_list)).astype('int')
        blue = red[::-1]
        rgb = [(r, 0, blue[i]) for i, r in enumerate(red)]
        out = ['#%02x%02x%02x' % val for val in rgb]
        if reverse:
            out = out[::-1]
        return out

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

    ################################
    # XML io
    ################################


    def save_xml(self, file_name=None, folder=None):
        """
        Save study to an xml file
        :param folder:
        :param file_name:
        :return:
        """

        if not file_name:
            import time
            file_name = '_'.join([time.strftime("%Y%m%d"), 'RockPy', self.name,
                                  '[{}]SG_[{}]S'.format(len(self.groupnames), len(self.samplelist))]) + '.rpy.xml'
        if not folder:
            folder = RockPy3.core.file_operations.default_folder
        log.info('SAVING RockPy data as XML to {}'.format(os.path.join(folder, file_name)))

        # create root node that contains studies
        root = etree.Element(Study.ROCKPY, attrib={'rockpy_revision': RockPy3.rev, 'rockpy_file_version': '0.1'})
        # append etree from this study to the root element
        root.append(self.etree)
        et = etree.ElementTree(root)

        et.write(os.path.join(folder, file_name))

    @property
    def etree(self):
        """
        Returns the content of the samplegroup as an xml.etree.ElementTree object which can be used to construct xml
        representation

        Returns
        -------
             etree: xml.etree.ElementTree
        """

        study_node = etree.Element(Study.STUDY, attrib={Study.NAME: str(self.name)})

        # create list of samples
        for s in self.samplelist:
            study_node.append(s.etree)

        return study_node

    @classmethod
    def from_etree(cls, et):
        if et.tag != Study.ROCKPY:
            log.ERROR("root tag must be {}".format(Study.ROCKPY))
            return None

        # #find first study - ignoring others
        s = et.find(Study.STUDY)
        if s is None:
            log.ERROR("no study found.")
            return None

        name = s.attrib[Study.NAME]
        log.info("reading study {}".format(name))

        study = cls(name=name)

        # readin the samples
        for sample_node in s.findall(RockPy3.core.sample.Sample.SAMPLE):
            study.add_sample(sobj=RockPy3.core.sample.Sample.from_etree(sample_node))

        return study

    @classmethod
    def load_from_xml(cls, file_name, folder=None):
        if not folder:
            folder = RockPy3.core.file_operations.default_folder

        log.info("reading xml data from {}".format(join(folder, file_name)))

        tree = etree.parse(join(folder, file_name))
        root = tree.getroot()

        return cls.from_etree(root)

if __name__ == '__main__':
    S400J = RockPy3.RockPyStudy()
    S.import_folder('/Users/mike/Dropbox/experimental_data/FeNiX/FeNi20K');
    S400J.info()