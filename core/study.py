import time

import tabulate

import RockPy3
from core import utils
import RockPy3.core.sample


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
        # self.log = log  # logging.getLogger('RockPy.' + type(self).__name__)
        RockPy3.logger.info('CREATING study << {} >>'.format(name))
        self.name = name
        self._samples = dict()  # {'sname':'sobj'}
        self._all_samplegroup = None

    def __repr__(self):
        if self == RockPy3.Study:
            return '<< RockPy.MasterStudy >>'.format(self.name)
        else:
            return '<< RockPy.Study.{} >>'.format(self.name)

    @property
    def samplelist(self):
        """
        Returns a list of all samples

        Returns
        -------
            list of all samples
        """

        return [v for k, v in self._samples.items()]

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

    def add_mean_samplegroup(self):
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

        if not any(i for i in locals() if i != 'self'):
            return slist

        # samplegroup filtering
        if gname:
            gname = utils.to_list(gname)
            slist = [s for s in slist if any(sg in gname for sg in s._samplegroups)]

        # sample filtering
        if sname:
            sname = utils.to_list(sname)
            slist = [s for s in slist if s.name in sname]

        return slist

    def get_samplegroup(self, gname):
        """
        wrapper for simply getting all samples of one samplegroup
        :param gname: str
            name of the samplegroup
        :return: list
            list of samples in that group
        """
        return self.get_sample(gname=gname)

    def get_measurement(self):
        pass

    # todo Python3
    def info(self, tablefmt='simple'):
        formats = ['plain', 'simple', 'grid', 'fancy_grid', 'pipe', 'orgtbl', 'rst', 'mediawiki', 'html', 'latex',
                   'latex_booktabs']

        if not tablefmt in formats:
            RockPy3.logger.info('NO SUCH FORMAT')
            tablefmt = 'simple'

        header = ['Sample Name', 'Sample Group', 'Measurements', 'series', 'Initial State']
        table = []

        for s in sorted(self.samplelist):
            mtypes = [m.mtype for m in s.measurements]
            stypes = sorted(list(set([stype for m in s.measurements for stype in m.stypes])))
            measurements = ', '.join(['%ix %s' % (mtypes.count(i), i) for i in sorted(set(mtypes))])
            stypes = ', '.join(stypes)
            i_state = [True if any(m.has_initial_state for m in s.measurements) else False][0]
            line0 = [s.name, s._samplegroups, measurements, stypes, i_state]
            table.append(line0)
            table.append([''.join(['--' for i in str(j)]) for j in line0])
            for m in s.measurements:
                if not isinstance(m, RockPy3.Packages.Generic.Measurements.parameters.Parameter):
                    if m.has_initial_state:
                        initial = m.initial_state.mtype
                    else:
                        initial = ''
                    line = ['', s.name, m.mtype,
                            ', '.join(['{} [{}]'.format(series[0], series[1]) for series in m.stype_sval_tuples]),
                            initial]
                    table.append(line)
            table.append([''.join(['--' for i in str(j)]) for j in line0])

        print(tabulate(table, headers=header, tablefmt=tablefmt))
