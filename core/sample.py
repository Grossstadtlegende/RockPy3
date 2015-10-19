import logging
import itertools
from copy import deepcopy
import RockPy3
import RockPy3.core.study
import core.utils


class Sample(object):
    snum = 0

    @property
    def log(self):
        return core.utils.set_get_attr(self, '_log',
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
                 samplegroup=None,
                 study=None,
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

        if not study:
            study = RockPy3.Study
        else:
            if not isinstance(study, RockPy3.core.study.Study):
                self.log.error('STUDY not a valid RockPy3.core.Study object. Using RockPy Masterstudy')
                study = RockPy3.Study


        self.name = name  # unique name, only one per study
        self.comment = comment

        # add sample to study
        self.study = study
        self.study.add_sample(sobj=self)
        self.idx = self.study.n_samples

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

        # adding paraeter measurements
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
            fpath=None, ftype=None,  # file path and file type
            idx=None,
            mdata=None,
            mobj=None,  # for special import of a measurement instance
            series=None,
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

        Returns
        -------
            RockPy3.measurement object
        '''

        # lookup abbreviations of mtypes and ftypes
        import_info = {}
        if mtype and ftype:
            mtype = mtype.lower()
            if mtype in RockPy3.mtype_ftype_abbreviations_inversed:
                mtype = RockPy3.mtype_ftype_abbreviations_inversed[mtype]
                import_info.setdefault('mtype', mtype)

            ftype = ftype.lower()
            if ftype in RockPy3.mtype_ftype_abbreviations_inversed:
                ftype = RockPy3.mtype_ftype_abbreviations_inversed[ftype]
                import_info.setdefault('ftype', ftype)

        if idx is None:
            idx = len(self.measurements)

        # data import from file
        if all([mtype, fpath, ftype]) or fpath:
            import_info = self.generate_import_info(mtype, fpath, ftype, idx, series)
            # if given add samplegroup to sample
            sg = import_info.pop('samplegroup', None)

            if sg:
                self.add_to_samplegroup(gname=sg)

            mtype = import_info.pop('mtype', mtype)

            if not self.mtype_not_implemented_check(mtype=mtype):
                return
            mobj = RockPy3.implemented_measurements[mtype].from_file(sobj=self, **import_info)

        # if mdata provided
        if all([mdata, mtype]):
            if not self.mtype_not_implemented_check(mtype=mtype):
                return
            mobj = RockPy3.implemented_measurements[mtype](sobj=self, mdata=mdata, series=series, idx=idx)

        if mobj:
            self.log.info('ADDING\t << %s, %s >>' % (import_info['ftype'], mtype))
            if series:
                self.log.info(
                    '\t\t WITH series << %s >>' % ('; '.join(', '.join(str(j) for j in i) for i in series)))
            if mobj not in self.measurements:
                self.measurements.append(mobj)
                self.raw_measurements.append(deepcopy(mobj))
                self._add_m2_mdict(mobj)
            return mobj
        else:
            self.log.error('COULD not create measurement << %s >>' % mtype)

    def add_simulation(self, mtype, idx=None, **sim_param):
        """
        add simulated measurements

        Parameters
        ----------
           mtype: str - the type of simulated measurement
           idx:
           sim_param: dict of parameters to specify simulation
        :return: RockPy.measurement object
        """
        mtype = mtype.lower()
        if mtype in RockPy3.mtype_ftype_abbreviations_inversed:
            mtype = RockPy3.mtype_ftype_abbreviations_inversed[mtype]

        if idx is None:
            idx = len(self.measurements)  # if there is no measurement index

        if mtype in RockPy3.implemented_measurements:
            mobj = RockPy3.implemented_measurements[mtype].from_simulation(sobj=self, idx=idx, **sim_param)
            if mobj:
                self.add_measurement(mtype=mtype, ftype='simulation', mobj=mobj)
                return mobj
            else:
                self.log.info('CANT ADD simulated measurement << %s >>' % mtype)
                return None
        else:
            self.log.error(' << %s >> not implemented, yet' % mtype)

    def add_mean_measurements(self,
                              interpolate=True, substfunc='mean', mean_of_mean=False,
                              reference=None, ref_dtype='mag', norm_dtypes='all', vval=None, norm_method='max',
                              normalize_variable=False, dont_normalize=None,
                              ignore_series=False):
        """
        Creates mean measurements for all measurements and measurement types
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
        # separate the different mtypes
        for mtype in self.mdict['mtype']:
            # all measurements with that mtype
            measurements = self.mdict['mtype'][mtype]
            # use first measurement as template to check for series
            while measurements:
                m = measurements[0]
                if isinstance(m, RockPy3.Packages.Generic.Measurements.parameters.Parameter):
                    break
                if ignore_series:
                    mlist = measurements
                else:
                    # get measurements with the same series
                    mlist = [measurement for measurement in measurements if m.equal_series(measurement)]
                # remove the measurements from the measurements list so they dont get averaged twice
                measurements = [m for m in measurements if m not in mlist]

                if not mlist:
                    break

                if self.mean_measurement_exists(mlist):
                    self.logger.warning('MEAN measurement already exists for these measurements:\n\t\t{}'.format(mlist))
                    break

                self.create_mean_measurement(mlist=mlist,
                                             ignore_series=ignore_series,
                                             interpolate=interpolate, substfunc=substfunc,
                                             reference=reference, ref_dtype=ref_dtype, norm_dtypes=norm_dtypes,
                                             vval=vval,
                                             norm_method=norm_method,
                                             normalize_variable=normalize_variable, dont_normalize=dont_normalize)

    def create_mean_measurement(self,
                                mtype=None, stype=None, sval=None, sval_range=None, series=None, invert=False,
                                mlist=None,
                                interpolate=False, substfunc='mean',
                                ignore_series=False,
                                recalc_mag=False,
                                reference=None, ref_dtype='mag', norm_dtypes='all', vval=None, norm_method='max',
                                normalize_variable=False, dont_normalize=None,
                                create_only=False):

        """
        takes a list of measurements and creates a mean measurement out of all measurements data

        Parameters
        ----------
            mtype: str
              mtype to be returned
            serie: list(tuple)
              list of tuples, to search for several sepcific series. e.g. [('mtime',4),('gc',2)] will only return
              mesurements that fulfill both criteria.
              Supercedes stype, sval and sval_range. Returnes only measurements that meet series exactly!
            stype: str
              series type
            sval_range: list, str
              series range e.g. sval_range = [0,2] will give all from 0 to 2 including 0,2
              also '<2', '<=2', '>2', and '>=2' are allowed.
            sval: float
              series value to be searched for.
              caution:
                 will be overwritten when sval_range is given
            invert:
              if invert true it returns only measurements that do not meet above criteria
            sval_range:
              can be used to look up measurements within a certain range. if only one value is given,
                     it is assumed to be an upper limit and the range is set to [0, sval_range]
            interpolate: bool
            substfunc: str
            recalc_mag: bool
            reference: str
            ref_dtype: str
            norm_dtypes: list
            vval: float
            norm_method: str
            normalize_variable: bool
            dont_normalize: list
            create_only: bool
                will not add measurement to the mean_measurements list or mean_mdict

        Returns
        -------
           RockPy.Measurement
              The mean measurement that fits to the specified lookup


        """
        # check for mtype if no mtype specified
        if mlist and not mtype:
            mtype = list(set(m.mtype for m in mlist))
            if len(mtype) != 1:
                raise TypeError('NO mtype specified. List of measurements may only contain one mtype')
            else:
                mtype = mtype[0]

        if not mtype and not mlist:
            raise TypeError('NO mtype specified. Please specify mtype')

        if not mlist:
            mlist = self.get_measurement(mtype=mtype, serie=series,
                                         stype=stype,
                                         sval=sval, sval_range=sval_range,
                                         mean=False, invert=invert)
        # normalze all measurements
        if reference:
            mlist = [m.normalize(reference=reference, ref_dtype=ref_dtype, norm_dtypes=norm_dtypes, vval=vval,
                                 norm_method=norm_method, normalize_variable=normalize_variable,
                                 dont_normalize=dont_normalize) for m in mlist]

        if not mlist:
            self.log.warning('NO measurement found >> %s, %s, %f >>' % (mtype, stype, sval))
            return None

        if len(mlist) == 1:
            self.log.warning('Only one measurement found returning measurement')
            # add to self.mean_measurements if specified
            if not create_only:
                self.mean_measurements.append(mlist[0])
                self._add_m2_mdict(mobj=mlist[0], mdict_type='mean_mdict')
            return mlist[0]

        mlist = [deepcopy(m) for m in mlist]  # create deepcopies #todo works?

        mean = RockPy3.implemented_measurements[mtype].from_measurements_create_mean(
            sobj=self, mlist=mlist, interpolate=interpolate, recalc_mag=recalc_mag,
            substfunc=substfunc, ignore_series=ignore_series)

        # add to self.mean_measurements if specified
        if not create_only:
            self.mean_measurements.append(mean)
            self._add_m2_mdict(mobj=mean, mdict_type='mean_mdict')
        return mean

    def add_to_samplegroup(self, gname):
        if gname not in self._samplegroups:
            self.log.debug('ADDING {} to samplegroup {}'.format(self.name, gname))
            self._samplegroups.append(gname)
        else:
            self.log.warning('SAMPLE {} already in samplegroup {}'.format(self.name, gname))

    def remove_from_samplegroup(self, gname):
        if gname in self._samplegroups:
            self.log.debug('REMOVING {} from samplegroup {}'.format(self.name, gname))
            self._samplegroups.remove(gname)

    def generate_import_info(self, mtype=None, fpath=None, ftype=None, idx=None, series=None):
        """
        First generate the file info. It is read from the fname, if possible.
        After that the mtype, ftype, fpath and idx will be overwritten, assuming that you gave a proper
        filname.
        """
        out = {'mtype':mtype, 'fpath':fpath, 'ftype':ftype, 'idx':idx, 'series':series}

        file_info = dict()
        with RockPy3.ignored(ValueError):
            file_info = RockPy3.core.file_operations.get_info_from_fname(path=fpath)
        if not file_info:
            self.log.warning(
                'CANNOT readin fpath automatically. See RockPy naming conventions for proper naming scheme.')
            fname = RockPy3.get_fname_from_info(samplegroup='SG', sample_name=self.name, mtype=mtype, ftype=ftype)
            self.log.info('FILE NAME proposition:')
            self.log.info('-'.join('' for i in range(50)))
            self.log.info('%s' % fname)
            self.log.info('-'.join('' for i in range(50)))
            self.log.info('Please check for consistency and add samplegroup and mass, height, diameter')
        else:
            for check in ['mtype', 'ftype', 'series']:
                if check in file_info and locals()[check]:
                    if not out[check] == file_info[check]:
                        self.log.warning('!!! INPUT != file_name: info does not match. Please check input, assuming filename correct')
                        self.log.warning('!!! {} != {}'.format(locals()[check], file_info[check]))
        out.update(file_info)
        out.pop('name', None)
        return out

    def get_mobj_from_file(self, mtype=None, fpath=None, ftype=None, idx=None, series=None):


        return mobj, file_info

    def mtype_not_implemented_check(self, mtype):
        if mtype not in RockPy3.implemented_measurements:
            self.log.error(' << %s >> not implemented, yet' % mtype)
            self.log.error('\tIMPLEMENTED: %s' % list(RockPy3.implemented_measurements.keys()))
            return
        else:
            return True

    """
    ####################################################################################################################
    PICKL
    """

    def __setstate__(self, dict):
        self.__dict__.update(dict)

    def __getstate__(self):
        """
        returned dict will be pickled
        :return:
        """
        pickle_me = {k: v for k, v in self.__dict__.items() if k in
                     ('name', 'index', 'color',
                      'measurements',
                      '_filtered_data', '_sample_groups',
                      '_mdict', '_mean_mdict',
                      'is_mean', 'mean_measurements', '_mean_results',
                      'results',
                      )}
        return pickle_me

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
                        series=None,
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
        if not any(i for i in [mtype, series, stype, sval, sval_range, mean]):
            if not invert:
                return self.measurements
            else:
                return []

        mtype = RockPy3.core.utils.to_list(mtype)
        mtype = [RockPy3.abbrev_to_name(mt) for mt in mtype]

        stype = RockPy3.core.utils.to_list(stype)
        sval = RockPy3.core.utils.to_list(sval)

        if mean:
            mdict = self.mean_mdict
            mdict_type = 'mean_mdict'
        else:
            mdict = self.mdict
            mdict_type = 'mdict'

        if sval_range:
            sval = self._convert_sval_range(sval_range=sval_range, mean=mean)
            self.log.info('SEARCHING %s for sval_range << %s >>' % (mdict_type, ', '.join(map(str, sval))))

        out = []

        if not series:
            for mt in mtype:
                for st in stype:
                    for sv in sval:
                        measurements = [m for m in mdict['measurements'] if
                                        m.has_mtype_stype_sval(mtype=mt, stype=st, sval=sv) if m not in out]
                        out.extend(measurements)
        else:
            # searching for specific series, all mtypes specified that fit the series description will be returned
            serie = RockPy3.utils.general.tuple2list_of_tuples(series)
            for mtype in mtype:  # cycle through mtypes
                aux = []
                for s in serie:
                    aux.extend(self.get_mtype_stype_sval(mtype=mt, stype=s[0], sval=float(s[1])))
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

        for k0, v0 in sorted(mdict.items()):
            if isinstance(v0, dict):
                for k1, v1 in sorted(v0.items()):
                    if isinstance(v1, dict):
                        for k2, v2 in sorted(v1.items()):
                            if isinstance(v2, dict):
                                for k3, v3 in sorted(v2.items()):
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
            self.log.info('SERIES & MEASURMENT << {}, {} >> already in mdict'.format(series, mobj))
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
