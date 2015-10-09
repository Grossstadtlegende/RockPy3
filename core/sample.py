import logging
import itertools
from copy import deepcopy
import RockPy3
import RockPy3.core.study
import core.utils


class Sample(object):
    snum = 0

    @property
    def logger(self):
        return core.utils.set_get_attr(self, '_logger',
                                       value=logging.getLogger('RockPy3.Sample(#%03i)[%s]' % (Sample.snum, self.name)))

    def __lt__(self, other):
        return self.name < other.name

    def __init__(self,
                 name=None,
                 comment='',
                 mass=None, mass_unit='kg', mass_ftype='generic',
                 height=None, diameter=None,
                 x_len=None, y_len=None, z_len=None,  # for cubic samples
                 length_unit='mm', length_ftype='generic',
                 sample_shape='cylinder',
                 coord=None,
                 samplegroup=None
                 ):

        """
        Parameters
        ----------
           name: str
              name of the sample.
           mass: float
              mass of the sample. If not kg then please specify the mass_unit. It is stored in kg.
           mass_unit: str
              has to be specified in order to calculate the sample mass properly.
           height: float
              sample height - stored in 'm'
           diameter: float
              sample diameter - stored in 'm'
           length_unit: str
              if not 'm' please specify
           length_machine: str
              if not 'm' please specify
           sample_shape: str
              needed for volume calculation
              cylinder: needs height, diameter
              cube: needs x_len, y_len, z_len
              sphere: diameter
           coord: str
              coordinate system
              can be 'core', 'geo' or 'bed'
           color: color
              used in plots if specified
        """
        if not name:
            name = 'S%02i' % Sample.snum
        self.name = name  # unique name, only one per study
        self.comment = comment
        self.idx = Sample.snum

        Sample.snum += 1

        self._samplegroups = []
        if samplegroup:
            self.add_to_samplegroup(gname=samplegroup)

        # coordinate system
        self._coord = coord

        RockPy3.logger.info('CREATING\t new sample << %s >>' % self.name)

        self.raw_measurements = []
        self.measurements = []
        self.results = None

        self.mean_measurements = []
        self._mean_results = None

        # dictionaries
        self._mdict = self._create_mdict()
        self._mean_mdict = self._create_mdict()
        self._rdict = self._create_mdict()

        if mass is not None:
            self.add_measurement(mtype='mass', fpath=None, ftype=mass_ftype,
                                        value=float(mass), unit=mass_unit)
        if diameter is not None:
            self.add_measurement(mtype='diameter', fpath=None, ftype=length_ftype,
                                            diameter=float(diameter), length_unit=length_unit)
        if height is not None:
            self.add_measurement(mtype='height', fpath=None, ftype=length_ftype,
                                          height=float(height), length_unit=length_unit)

        if x_len:
            x_len = self.add_measurement(mtype='length', fpath=None, ftype=length_ftype,
                                         value=float(x_len), unit=length_unit, direction='x')
        if y_len:
            y_len = self.add_measurement(mtype='length', fpath=None, ftype=length_ftype,
                                         value=float(y_len), unit=length_unit, direction='y')
        if z_len:
            z_len = self.add_measurement(mtype='length', fpath=None, ftype=length_ftype,
                                         value=float(z_len), unit=length_unit, direction='z')

        if height and diameter:
            self.add_measurement(mtype='volume', sample_shape=sample_shape, height=height, diameter=diameter)
        if x_len and y_len and z_len:
            self.add_measurement(mtype='volume', sample_shape=sample_shape, x_len=x_len, y_len=y_len, z_len=z_len)

    def __repr__(self):
        return '<< RockPy3.Sample.{} >>'.format(self.name)

    ####################################################################################################################
    ''' class methods '''

    ####################################################################################################################
    ''' ADD FUNCTIONS '''

    def add_measurement(
            self,
            mtype=None,  # measurement type
            fpath=None, ftype='generic',  # file path and file type
            idx=None,
            mdata=None,
            # for special import of pure data (needs to be formatted as specified in data of measurement class)
            mobj=None,  # for special import of a measurement instance
            series=None,
            create_parameter=None,  # todo implement
            create_only=False,
            **options):
        '''
        All measurements have to be added here

        Parameters
        ----------
        mtype: str
          the type of measurement
          default: None

        fpath: str
          the complete path to the measurement file
          default: None

        ftype: str
          the filetype from which the file is output
          default: 'generic'

        idx: index of measurement
          default: None, will be the index of the measurement in sample.measurements

        mdata: any kind of data that must fit the required structure of the data of the measurement
            will be used instead of data from file
            example:
                mdata = dict(mass=10)
                mdata = dict( x=[1,2,3,4], y = [1,2,3,4],...)

        create_parameter: bool NOT IMPLEMENTED YET
            if true it will create the parameter (lenght, mass) measurements from path
            before creating the actual measurement
            default: False
            Note: NEEDS PATH with RockPy complient fname structure.
        mobj: RockPy3.Measurement object
            if provided, the object is added to self.measurements
        create_only: bool
            if True the measurement is only created and not appended to self.measurements, self.raw_measurements
            or any mdict


        Returns
        -------
            RockPy3.measurement object

        :mtypes:

        - mass
        '''
        # abbrev, inv_abbrev = RockPy3.core.file_operations.mtype_ftype_abbreviations()
        abbrev, i_abbrev = RockPy3.mtype_ftype_abbreviations, RockPy3.mtype_ftype_abbreviations_inversed

        ### FILE IMPORT
        file_info = {}  # file_info includes all info needed for creation of measurement instance
        auto_import = False  # flag for filename creation
        # create an index for the measurement if none is provided
        if idx is None:
            idx = len(self.measurements)

        # if auomatic import through filename is needed:
        # either fname AND folder are given OR the full path is passed
        # then the file_info dictionary is created
        if fpath:
            try:
                file_info = RockPy3.core.file_operations.get_info_from_fname(path=fpath)
                file_info.update(dict(sobj=self))
                auto_import = True
            except (KeyError, IndexError, ValueError):
                # if file_info can not be generated from the pathname
                # mtype AND ftype have to exist otherwise measurement can not be created
                if not mtype or not ftype:
                    self.logger.error(
                        'NO mtype and/or ftype specified, cannot readin fpath automatically.')
                else:
                    self.logger.warning(
                        'CANNOT readin fpath automatically. See RockPy naming conventions for proper naming scheme.')
                    fname = RockPy3.get_fname_from_info(samplegroup='SG', sample_name=self.name, mtype=mtype,
                                                        ftype=ftype)
                    self.logger.info('FILE NAME proposition:')
                    self.logger.info('')
                    self.logger.info('                   %s' % fname)
                    self.logger.info('')
                    self.logger.info('Please check for consistency and add samplegroup and mass, height, diameter')
        # create the file_info dictionary for classic import
        # fpath not necessarily needed here, some measurements (e.g. Parameter.daughters)
        # dont need a file because the data is directly added
        if mtype and ftype:
            mtype = mtype.lower()
            if mtype in i_abbrev:
                mtype = i_abbrev[mtype]

            ftype = ftype.lower()
            if ftype in i_abbrev:
                ftype = i_abbrev[ftype]

            file_info = dict(sobj=self,
                             mtype=mtype, fpath=fpath, ftype=ftype,
                             m_idx=idx, mdata=mdata)

        # add the additional series information to the file_info dict
        if series:
            # check if single series
            # workaround because list(tuple(A)) -> list(A) not list(tuple(A))???
            if type(series) == tuple:
                aux = list()
                aux.append(series)
                series = aux
            file_info.update(dict(series=series))

        if options:
            file_info.update(options)

        # all data is now stored in the file_ino dictionary and can be used to call the constructor of the
        # measurement class
        # NOTE: if mobj is passed
        if file_info or mobj:
            # check if mtype exists of instance is present
            mtype = file_info.get('mtype', None)

            if mtype in RockPy3.implemented_measurements or mobj:
                self.logger.info('ADDING\t << measurement >> %s' % mtype)
                if series:
                    self.logger.info(
                        '\t\t WITH series << %s >>' % ('; '.join(', '.join(str(j) for j in i) for i in series)))
                if mobj:
                    measurement = mobj  # todo mobj.sobj = self?
                else:
                    # create instance from implemented_measurements dictionary
                    # call constructor of a subclassed measurement
                    measurement = RockPy3.implemented_measurements[mtype](**file_info)
                if measurement and not create_only:
                    self.measurements.append(measurement)
                    # append a deepcopy of the measurement to the raw_measurement list so that it is possible to
                    # undo any calculations
                    # self.raw_measurements.append(deepcopy(measurement))
                    # if measurement has series, it is added to the mdict automatically, if not we have to add it here
                    # in this case the measurement has only one series and its type must be 'none'
                    if len(measurement.series) == 1 and measurement.series[0].stype == 'none':
                        self._add_m2_mdict(measurement)
                    return measurement
                if create_only:
                    # measurement gets added to sample_mdict in the constructor of the measurement, therfore it would still be in the mdict if it is only created. We have to remove it again
                    self._remove_m_from_mdict(measurement)
                    return measurement
                else:
                    return None
            else:
                self.logger.error(' << %s >> not implemented, yet' % mtype)
                self.logger.error('\tIMPLEMENTED: %s' % RockPy3.implemented_measurements.keys())

    def add_to_samplegroup(self, gname):
        self.logger.debug('ADDING {} to samplegroup {}'.format(self.name, gname))
        self._samplegroups.append(gname)

    def remove_from_samplegroup(self, gname):
        self.logger.debug('REMOVING {} from samplegroup {}'.format(self.name, gname))
        self._samplegroups.remove(gname)


    ####################################################################################################################
    ''' GET FUNCTIONS '''
    def _convert_sval_range(self, sval_range, mean):
        """
        converts a string of svals into a list

        Parameters
        ----------
            sval_range: list, str
                series range e.g. sval_range = [0,2] will give all from 0 to 2 including 0,2
                also '<2', '<=2', '>2', and '>=2' are allowed.

        """

        if mean:
            mdict = self.mean_mdict
        else:
            mdict = self.mdict

        if isinstance(sval_range, list):
            svals = [i for i in mdict['sval'] if sval_range[0] <= i <= sval_range[1]]
        if isinstance(sval_range, str):
            sval_range = sval_range.strip()  # remove whitespaces in case '> 4' is provided
            if '<' in sval_range:
                if '=' in sval_range:
                    svals = [i for i in mdict['sval'] if i <= float(sval_range.replace('<=', ''))]
                else:
                    svals = [i for i in mdict['sval'] if i < float(sval_range.replace('<', ''))]
            if '>' in sval_range:
                if '=' in sval_range:
                    svals = [i for i in mdict['sval'] if i >= float(sval_range.replace('>=', ''))]
                else:
                    svals = [i for i in mdict['sval'] if i > float(sval_range.replace('>', ''))]
        return sorted(svals)

    def get_measurement(self,
                         mtype=None,
                         serie=None,
                         stype=None, sval=None, sval_range=None,
                         mean=False,
                         invert=False,
                         ):
        """
        Returns a list of measurements of type = mtypes

        Parameters
        ----------
           mtypes: list, str
              mtypes to be returned
           series: list(tuple)
              list of tuples, to search for several sepcific series. e.g. [('mtime',4),('gc',2)] will only return
              mesurements that fulfill both criteria.
              Supercedes stype, sval and sval_range. Returnes only measurements that meet series exactly!
           stypes: list, str
              series type
           sval_range: list, str
              series range e.g. sval_range = [0,2] will give all from 0 to 2 including 0,2
              also '<2', '<=2', '>2', and '>=2' are allowed.
           svals: float
              series value to be searched for.
              caution:
                 will be overwritten when sval_range is given
           invert:
              if invert true it returns only measurements that do not meet criteria
           sval_range:
              can be used to look up measurements within a certain range. if only one value is given,
                     it is assumed to be an upper limit and the range is set to [0, sval_range]
           mean: bool

        Returns
        -------
            if no arguments are passed all sample.measurements
            list of RockPy.Measurements that meet search criteria or if invert is True, do not meet criteria.
            [] if none are found

        Note
        ----
            there is no connection between stype and sval. This may cause problems. I you have measurements with
               M1: [pressure, 0.0, GPa], [temperature, 100.0, C]
               M2: [pressure, 1.0, GPa], [temperature, 100.0, C]
            and you search for stypes=['pressure','temperature'], svals=[0,100]. It will return both M1 and M2 because
            both M1 and M2 have [temperature, 100.0, C].

        """
        # if no parameters are given, return all measurments/none (invert=True)
        if not any(i for i in [mtype, serie, stype, sval, sval_range, mean]):
            if not invert:
                return self.measurements
            else:
                return []

        mtype = RockPy3.utils.to_list(mtype)
        mtype = [RockPy3.abbrev_to_name(mtype) for mtype in mtypes]

        stype = to_list(stype)
        sval = to_list(sval)

        if mean:
            mdict = self.mean_mdict
            mdict_type = 'mean_mdict'
        else:
            mdict = self.mdict
            mdict_type = 'mdict'

        if sval_range:
            sval = self._convert_sval_range(sval_range=sval_range, mean=mean)
            self.logger.info('SEARCHING %s for sval_range << %s >>' % (mdict_type, ', '.join(map(str, sval))))

        out = []

        if not serie:
            for mtype in mtype:
                for stype in stype:
                    for sval in sval:
                        measurements = [m for m in mdict['measurements'] if
                                        m.has_mtype_stype_sval(mtype=mtype, stype=stype, sval=sval) if m not in out]
                        out.extend(measurements)
        else:
            # searching for specific series, all mtypes specified that fit the series description will be returned
            serie = RockPy.utils.general.tuple2list_of_tuples(serie)
            for mtype in mtype:  # cycle through mtypes
                aux = []
                for s in serie:
                    aux.extend(self.get_mtype_stype_sval(mtype=mtype, stype=s[0], sval=float(s[1])))
                out.extend(list(set([i for i in aux if aux.count(i) == len(series)])))

        # invert list to contain only measurements that do not meet criteria
        if invert:
            out = [i for i in mdict['measurements'] if not i in out]
        return out

    ####################################################################################################################
    ''' MEASUREMENT / RESULT DICTIONARY PART'''

    @property
    def mdict(self):
        """
        """
        if not self._mdict:
            self._mdict = self._create_mdict()
        else:
            return self._mdict

    @property
    def mean_mdict(self):
        if not self._mean_mdict:
            self._mean_mdict = self._create_mdict()
        else:
            return self._mean_mdict

    def _create_mdict(self):
        """
        creates all info dictionaries

        Returns
        -------
           dict
              Dictionary with a permutation of ,type, stype and sval.
        """
        d = ['mtype', 'stype', 'sval']
        keys = ['_'.join(i) for n in range(4) for i in itertools.permutations(d, n) if not len(i) == 0]
        out = {i: {} for i in keys}
        out.update({'measurements': list()})
        out.update({'series': list()})
        return out

    def _mdict_cleanup(self, mdict_type='mdict'):
        """
        recursively removes all empty lists from dictionary
        :param empties_list:
        :return:
        """

        mdict = getattr(self, mdict_type)

        for k0, v0 in sorted(mdict.iteritems()):
            if isinstance(v0, dict):
                for k1, v1 in sorted(v0.iteritems()):
                    if isinstance(v1, dict):
                        for k2, v2 in sorted(v1.iteritems()):
                            if isinstance(v2, dict):
                                for k3, v3 in sorted(v2.iteritems()):
                                    if not v3:
                                        v2.pop(k3)
                                    if not v2:
                                        v1.pop(k2)
                                    if not v1:
                                        v0.pop(k1)
                            else:
                                if not v2:
                                    v1.pop(k2)
                                if not v1:
                                    v0.pop(k1)
                    else:
                        if not v1:
                            v0.pop(k1)

    def _add_m2_mdict(self, mobj, mdict_type='mdict'):
        """
        adds or removes a measurement from the mdict

        Parameters
        ----------
           mobj: measurement object
              object to be added
        :param operation:
        :return:
        """
        # cylcle through all the series
        for s in mobj.series:
            self._add_series2_mdict(mobj=mobj, series=s, mdict_type=mdict_type)

    def _remove_m_from_mdict(self, mobj, mdict_type='mdict'):
        """
        adds or removes a measurement from the mdict

        Parameters
        ----------
           mobj: measurement object
              object to be added
        :param operation:
        :return:
        """
        # cylcle through all the series
        for series in mobj.series:
            self._remove_series_from_mdict(mobj=mobj, series=series, mdict_type=mdict_type)

    def _add_series2_mdict(self, mobj, series, mdict_type='mdict'):
        self._change_series_in_mdict(mobj=mobj, series=series, operation='append', mdict_type=mdict_type)

    def _remove_series_from_mdict(self, mobj, series, mdict_type='mdict'):
        self._change_series_in_mdict(mobj=mobj, series=series, operation='remove', mdict_type=mdict_type)

    def _change_series_in_mdict(self, mobj, series, operation, mdict_type='mdict'):
        # dict for getting the info of the series
        sinfo = {'mtype': mobj.mtype, 'stype': series.stype, 'sval': series.value}

        mdict = getattr(self, mdict_type)

        if series in mdict['series'] and mobj in mdict['measurements'] and operation == 'append':
            self.logger.info('SERIES & MEASURMENT << {}, {} >> already in mdict'.format(series, mobj))
            return

        # cycle through all the elements of the self.mdict
        for level in mdict:
            # get sublevels of the level
            sublevels = level.split('_')
            if level == 'measurements':
                RockPy3.core.utils.append_if_not_exists(mdict['measurements'], mobj, operation=operation)
                # getattr(self.mdict['measurements'], operation)(mobj)
            elif level == 'series':
                RockPy3.core.utils.append_if_not_exists(mdict['series'], series, operation=operation)

                # getattr(self.mdict['series'], operation)(series)
            elif len(sublevels) == 1:
                d = mdict[level].setdefault(sinfo[level], list())
                RockPy3.core.utils.append_if_not_exists(d, mobj, operation=operation)
                # getattr(d, operation)(mobj)
            else:
                for slevel_idx, sublevel in enumerate(sublevels):
                    if slevel_idx == 0:
                        info0 = sinfo[sublevel]
                        d = mdict[level].setdefault(info0, dict())
                    elif slevel_idx != len(sublevels) - 1:
                        info0 = sinfo[sublevel]
                        d = d.setdefault(info0, dict())
                    else:
                        info0 = sinfo[sublevel]
                        d = d.setdefault(info0, list())
                        RockPy3.core.utils.append_if_not_exists(d, mobj, operation=operation)

                        # getattr(d, operation)(mobj)

        if operation == 'remove':
            self._mdict_cleanup(mdict_type=mdict_type)

    def _populate_mdict(self, mdict_type='mdict'):
        """
        Populates the mdict with all measurements
        :return:
        """
        if mdict_type == 'mdict':
            map(self._add_m2_mdict, self.measurements)
        if mdict_type == 'mean_mdict':
            add_m2_mean_mdict = partial(self._add_m2_mdict, mdict_type='mean_mdict')
            map(add_m2_mean_mdict, self.measurements)
