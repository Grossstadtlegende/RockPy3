import time
import RockPy3
from RockPy3.core import utils


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
        self.name = name
        self._samples = dict()  # {'sname':'sobj'}
        self._all_samplegroup = None

    def __repr__(self):
        return self.name

    @property
    def samplelist(self):
        return [v for k, v in self._samples.items()]

    @property
    def samplenames(self):
        return [k for k, v in self._samples.items()]


    ####################################################################################################################
    ''' add functions '''

    def _add_sample(self, sobj):
        sobj=utils.to_list(sobj)
        for s in sobj:
            self._samples.setdefault(s.name, s)

    def add_samplegroup(self, name=None):
        """
        creates a samplegroups and adds it to the samplegroup dictionary

        Parameter
        ---------
            name: str
            default: None
            if None, name is 'SampleGroup #samplegroups'

        Returns
        -------
            RockPy3.SampleGroup

        """
        sg = RockPy3.SampleGroup(name=name)
        self._samplegroups.setdefault(sg.name, sg)
        return sg

    def add_mean_samplegroup(self):
        pass

    ####################################################################################################################
    ''' remove functions '''

    def remove_samplegroup(self, name=None):
        pass

    ####################################################################################################################
    ''' get functions '''

    def get_sample(self):
        pass

    def get_measurement(self):
        pass

    # todo Python3
    def info(self, tablefmt='simple'):
        formats = ['plain', 'simple', 'grid', 'fancy_grid', 'pipe', 'orgtbl', 'rst', 'mediawiki', 'html', 'latex',
                   'latex_booktabs']

        if not tablefmt in formats:
            RockPy.logger.info('NO SUCH FORMAT')
            tablefmt = 'simple'

        header = ['Sample Group', 'Sample Name', 'Measurements', 'series', 'Initial State']
        table = []

        for sg in sorted(self._samplegroups):
            for sname, s in sorted(sg.sdict.iteritems()):
                mtypes = [m.mtype for m in s.measurements]
                stypes = sorted(list(set([stype for m in s.measurements for stype in m.stypes])))
                measurements = ', '.join(['%ix %s' % (mtypes.count(i), i) for i in sorted(set(mtypes))])
                stypes = ', '.join(stypes)
                i_state = [True if any(m.has_initial_state for m in s.measurements) else False][0]
                line0 = [sg.name, s.name, measurements, stypes, i_state]
                table.append(line0)
                table.append([''.join(['--' for i in str(j)]) for j in line0])
                for m in s.measurements:
                    if not isinstance(m, RockPy.Packages.Generic.Measurements.parameters.Parameter):
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
