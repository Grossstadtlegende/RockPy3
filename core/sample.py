import logging
import itertools
from copy import deepcopy
import RockPy3
import RockPy3.core.study
import RockPy3.core.utils
import numpy as np
from functools import partial
import xml.etree.ElementTree as etree

log = logging.getLogger(__name__)


class Sample(object):
    snum = 0

    SAMPLE = 'sample'
    SAMPLEGROUP = 'samplegroup'
    NAME = 'name'

    @property
    def log(self):
        return RockPy3.core.utils.set_get_attr(self, '_log',
                                               value=logging.getLogger(
                                                   'RockPy3.Sample(#%03i)[%s]' % (Sample.snum, self.name)))

    def __lt__(self, other):
        return self.name < other.name

    @property
    def samplegroups(self):
        if self._samplegroups:
            return self._samplegroups
        else:
            return ('None',)

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

        # self.raw_measurements = []
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
            mass = RockPy3.implemented_measurements['mass'](sobj=self,
                                                            mass=mass, mass_unit=mass_unit, ftype=mass_ftype)
            self.add_measurement(mobj=mass)
        if diameter is not None:
            diameter = RockPy3.implemented_measurements['diameter'](sobj=self,
                                                                    diameter=diameter, length_unit=length_unit,
                                                                    ftype=length_ftype)
            self.add_measurement(mobj=diameter)
        if height is not None:
            height = RockPy3.implemented_measurements['height'](sobj=self,
                                                                height=height, length_unit=length_unit,
                                                                ftype=length_ftype)
            self.add_measurement(mobj=height)

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
            create_parameter=False,
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

        create_parameter:
            default: True
            if true it will create the parameter (lenght, mass) measurements from path
            !!! before creating the actual measurement !!!
            Note: NEEDS PATH with RockPy complient fname structure.
        mobj: RockPy3.Measurement object
            if provided, the object is added to self.measurements

        Returns
        -------
            RockPy3.measurement object
        '''

        # lookup abbreviations of mtypes and ftypes
        import_info = {}
        import_info.update(options)
        import_info.update(dict(series=series))

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

        """
        ################################################################################################################
        # DATA import from FILE
        """
        if all([mtype, fpath, ftype]) or fpath:
            # check if we can read the filename - fails if not
            try:
                import_info = self.generate_import_info(mtype, fpath, ftype, idx, series)
            except:
                import_info = dict(mtype=mtype, ftype=ftype, fpath=fpath, series=series)

            # update any options e.g. series
            import_info.update(options)

            # remove the parameter_info from the info_dict
            parameter_info = {key: import_info.pop(key, None) for key in list(import_info.keys())
                              if key in ['mass', 'diameter', 'height',
                                         'x_len', 'y_len', 'z_len',
                                         'length_unit', 'mass_unit', 'volume']}

            # create the parameter measurement.
            if parameter_info and create_parameter:
                for mtype in ('mass', 'diameter', 'height'):
                    if mtype in parameter_info:
                        param = RockPy3.implemented_measurements[mtype](sobj=self, **parameter_info)
                        self._add_mobj(param)

            # if given add samplegroup to sample
            sg = import_info.pop('samplegroup', None)

            if sg:
                self.add_to_samplegroup(gname=sg)

            mtype = import_info.pop('mtype', mtype)

            if not self.mtype_not_implemented_check(mtype=mtype):
                return

            mobj = RockPy3.implemented_measurements[mtype].from_file(sobj=self, **import_info)

        """
        ################################################################################################################
        # DATA import from mass, height, diameter, len ...
        """

        if any(i in options for i in ['mass', 'diameter', 'height', 'x_len', 'y_len', 'z_len']):
            mobj = RockPy3.implemented_measurements[mtype](sobj=self, **options)

        """
        ################################################################################################################
        # DATA import from MDATA
        """
        if all([mdata, mtype]):
            if not self.mtype_not_implemented_check(mtype=mtype):
                return
            mobj = RockPy3.implemented_measurements[mtype](sobj=self, mdata=mdata, series=series, idx=idx)

        """
        ################################################################################################################
        # DATA import from MOBJ
        """
        if mobj:
            if isinstance(mobj, tuple) or ftype == 'from_measurement':
                if not self.mtype_not_implemented_check(mtype=mtype):
                    return
                mobj = RockPy3.implemented_measurements[mtype].from_measurement(sobj=self, mobj=mobj, **import_info)
            if not mobj:
                return
            self.log.info('ADDING\t << %s, %s >>' % (mobj.ftype, mobj.mtype))
            if series:
                self.log.info(
                    '\t\t WITH series << %s >>' % ('; '.join(', '.join(str(j) for j in i) for i in series)))
            self._add_mobj(mobj)
            return mobj
        else:
            self.log.error('COULD not create measurement << %s >>' % mtype)

    def _add_mobj(self, mobj):
        if mobj not in self.measurements:
            self.measurements.append(mobj)
            # self.raw_measurements.append(deepcopy(mobj))
            if mobj.is_mean:
                self._add_m2_mdict(mobj, mdict_type='mean_mdict')
            else:
                self._add_m2_mdict(mobj)

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
        mean_measurements = []
        # separate the different mtypes
        for mtype in self.mdict['mtype']:
            # all measurements with that mtype
            measurements = self.get_measurement(mtype=mtype, mean=False)  # self.mdict['mtype'][mtype]

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

                # remove the measurements from the measurements list so they don't get averaged twice
                measurements = [m for m in measurements if m not in mlist]

                if not mlist:
                    self.log.debug('NO MORE measurements in mlist')
                    break

                if self.mean_measurement_exists(mlist):
                    self.log.warning('MEAN measurement already exists for these measurements:\n\t\t{}'.format(mlist))
                    mean_measurements.extend(self.mean_measurement_exists(mlist))
                    continue

                mean_measurements.append(self.create_mean_measurement(mlist=mlist,
                                                                      ignore_series=ignore_series,
                                                                      interpolate=interpolate, substfunc=substfunc,
                                                                      reference=reference, ref_dtype=ref_dtype,
                                                                      norm_dtypes=norm_dtypes,
                                                                      vval=vval,
                                                                      norm_method=norm_method,
                                                                      normalize_variable=normalize_variable,
                                                                      dont_normalize=dont_normalize))
        return mean_measurements

    def create_mean_measurement(self,
                                mtype=None, stype=None, sval=None, sval_range=None, series=None, invert=False,
                                mlist=None,
                                interpolate=False, substfunc='mean',
                                ignore_series=False,
                                recalc_mag=False,
                                reference=None, ref_dtype='mag', norm_dtypes='all', vval=None, norm_method='max',
                                normalize_variable=False, dont_normalize=None,
                                create_only=False,
                                color=None, marker=None, linestyle=None):

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
            mlist = self.get_measurement(mtype=mtype, series=series,
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

        mtype = RockPy3.abbrev_to_classname(mtype)

        if len(mlist) == 1:
            self.log.warning('Only one measurement found returning measurement')
            # # add to self.mean_measurements if specified
            # if not create_only:
            #     mlist[0].base_measurements = mlist
            #     self.mean_measurements.append(mlist[0])
            #     self._add_m2_mdict(mobj=mlist[0], mdict_type='mean_mdict')
            # return mlist[0]

        # mlist = [m for m in mlist]  # create deepcopies

        # create mean measurement from a list of measurements
        mean = RockPy3.implemented_measurements[mtype].from_measurements_create_mean(
            sobj=self, mlist=mlist, interpolate=interpolate, recalc_mag=recalc_mag,
            substfunc=substfunc, ignore_series=ignore_series, color=color, marker=marker, linestyle=linestyle)

        # add to self.mean_measurements if specified
        if not create_only:
            self.mean_measurements.append(mean)
            self._add_m2_mdict(mobj=mean, mdict_type='mean_mdict')
        return mean

    def mean_measurement_exists(self, mlist):
        """
        Returns True if there is a mean measuement that contains all measurements in the mlist as its base measurements
        Now uses the unique id of any measurement object.
        """
        if not self.mean_measurements:
            return False
        id_list = set(m.id for m in mlist)
        mean = [mean for mean in self.mean_measurements if set(mean.base_ids) == id_list]
        return mean if mean else False

    def add_to_samplegroup(self, gname):
        gname = str(gname)
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
        out = {'mtype': mtype, 'fpath': fpath, 'ftype': ftype, 'idx': idx, 'series': series}

        file_info = dict()
        with RockPy3.ignored(ValueError):
            file_info = RockPy3.core.file_operations.get_info_from_fname(path=fpath)
        if not file_info:
            self.log.warning(
                'CANNOT readin fpath automatically. See RockPy naming conventions for proper naming scheme.')
            fname = RockPy3.get_fname_from_info(samplegroup='SG', sample_name=self.name, mtype=mtype, ftype=ftype,
                                                series=series)
            self.log.info('FILE NAME proposition:')
            self.log.info('-'.join('' for i in range(50)))
            self.log.info('%s' % fname)
            self.log.info('-'.join('' for i in range(50)))
            self.log.info('Please check for consistency and add samplegroup and mass, height, diameter')
        else:
            for check in ['mtype', 'ftype', 'series']:
                if check in file_info and locals()[check]:
                    if not out[check] == file_info[check]:
                        self.log.warning(
                            '!!! INPUT != file_name: info does not match. Please check input, assuming filename correct')
                        self.log.warning('!!! {} != {}'.format(locals()[check], file_info[check]))
        out.update(file_info)
        out.pop('name', None)
        return out

    # def get_mobj_from_file(self, mtype=None, fpath=None, ftype=None, idx=None, series=None):
    #
    #     return mobj, file_info

    def mtype_not_implemented_check(self, mtype):
        if mtype not in RockPy3.implemented_measurements:
            self.log.error(' << %s >> not implemented, yet' % mtype)
            self.log.error('\tIMPLEMENTED: %s' % list(RockPy3.implemented_measurements.keys()))
            return
        else:
            return True

    def in_samplegroup(self, gnames):
        """
        tests whether a sample is in ALL samplegroups specified
        """
        gnames = RockPy3.core.utils.to_list(gnames)
        if all(gname in self._samplegroups for gname in gnames):
            return True

    """
    ####################################################################################################################
    RESULTS
    """

    def get_result(self, result, mtype=None, mean=False, base=False, return_mean=True, **calculation_parameter):
        """
        A function that returns a list of all the result calculated from all measurements, that actually have the result

        Parameters
        ----------
            result: str:
                the result to be calculated
            mtype: str or list of str
                if provided, only results from that mtype are calculated
            calculation_parameter: dict
                the calculation parameters to be used
            mean: bool
                if true the results are calculated for the mean measuremnets
            base: bool
                if true the results are calculated for the base measurements (needs to be a mean measurement)
            return_mean: bool
                if true a numpy mean is calculated and returned


        Returns
        -------
            list of results

        Note
        ----
            each of the results is a tuple of the actual value and the error if calculated
        """
        # get all measurements that have the result
        # if mean use only mean measurements
        if mean:
            mlist = filter(lambda x: x.has_result(result=result), self.mean_measurements)
            if base:
                mlist = [m for mean in mlist for m in mean.base_measurements if m.has_result(result=result)]
        # if not use all non mean measurements
        else:
            mlist = filter(lambda x: x.has_result(result=result), self.measurements)

        if mtype:
            mtypes = RockPy3.to_list(mtype)
            mlist = filter(lambda x: x.mtype in mtypes, mlist)

        res = [getattr(m, 'result_' + result)(**calculation_parameter) for m in mlist]
        self.log.debug('Calculating result << {} >> for {} measurements'.format(result, len(res)))

        if return_mean:
            res = [(np.mean(res, axis=0)[0], np.std(res, axis=0)[0])]
        return res

    def set_recipe(self, result, recipe):
        """
        Sets a recipe for all measurements that have the result
        """
        mlist = self.get_measurement_new(result=result, all_types=True)
        self.log.debug('setting recipe = {} for result = {} for {} measuremnts'.format(recipe, result, len(mlist)))

        for m in mlist:
            m.set_recipe(result=result, recipe=recipe)

    def get_measurement_new(self, mtype=None, stype=None, sval=None, result=None, mean=False, base=False, all_types=False):

        if mean or all_types:
            mlist = self.mean_measurements
            if base or all_types:
                mlist = [m for mean in mlist for m in mean.base_measurements]
        else:
            mlist = self.measurements

        # if all types are supposed to be returned, hence the all_types boolean, mlost has to be extended
        if all_types:
            mlist.extend(self.measurements)
            mlist.extend(self.mean_measurements)

        if mtype:
            mtype = RockPy3.to_list(mtype)
            mlist = filter(lambda x: x.mtype in mtype, mlist)
        if stype:
            stype = RockPy3.to_list(stype)
            mlist = filter(lambda x: x.stype in stype, mlist)
        if result:
            mlist = filter(lambda x: x.has_result(result=result), mlist)

        return list(mlist)

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
                        id=None
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
           id: list(int)
            search for given measurement id

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

        """  # todo fix search for only series

        if id:
            id = RockPy3.core.utils.to_list(id)
            if not invert:
                m = [m for m in self.measurements + self.mean_measurements if m.id in id]
            else:
                m = [m for m in self.measurements + self.mean_measurements if m.id not in id]
            return m

        # if no parameters are given, return all measurments/none (invert=True)
        if not any(i for i in [mtype, series, stype, sval, sval_range]):
            if not invert:
                if mean:
                    return self.mean_measurements
                else:
                    return self.measurements
            else:
                return []

        if mtype:
            mtype = RockPy3.core.utils.to_list(mtype)
            mtype = [RockPy3.abbrev_to_classname(mt) for mt in mtype]
        else:
            mtype = [None]

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
                if not stype:
                    stype = [None]
                for st in stype:
                    if not sval:
                        sval = [None]
                    for sv in sval:
                        measurements = [m for m in mdict['measurements'] if
                                        m.has_mtype_stype_sval(mtype=mt, stype=st, sval=sv) if m not in out]
                        out.extend(measurements)
        else:
            # searching for specific series, all mtypes specified that fit the series description will be returned
            serie = RockPy3.core.utils.tuple2list_of_tuples(series)
            for mt in mtype:  # cycle through mtypes
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
            # var = lambda x: self._add_m2_mdict(x), self.measurements
            [self._add_m2_mdict(m) for m in self.measurements]
        if mdict_type == 'mean_mdict':
            add_m2_mean_mdict = partial(self._add_m2_mdict, mdict_type='mean_mdict')
            [add_m2_mean_mdict(m) for m in self.mean_measurements]

    def calc_all(self, **parameter):
        for m in self.measurements:
            m.calc_all(**parameter)

    @property
    def stypes(self):
        return list(self.mdict['stype'].keys())

    @property
    def mtypes(self):
        return list(self.mdict['mtype'].keys())

    ####################################################################################################################
    ''' PLOTTING PART'''

    def set_plt_prop(self, prop, value):
        for m in self.measurements + self.mean_measurements:
            m.set_plt_prop(prop, value)

    def series_to_color(self, stype):
        """
        adds corresponding color to each measurement with the stype
        """
        for m in self.measurements:
            if m.has_series(stype=stype.lower()):
                m.series_to_color(stype=stype.lower())

    ####################################################################################################################
    ''' LABES PART'''

    def label_add_sname(self):
        for m in self.measurements:
            m.label_add_sname()

    def label_add_series(self, stypes=None, add_stype=True, add_unit=True):
        self.label_add_stype(stypes=stypes, add_stype=add_stype, add_unit=add_unit)

    def label_add_stype(self, stypes=None, add_stype=True, add_unit=True):
        for m in self.measurements:
            m.label_add_stype(stypes=stypes, add_stype=add_stype, add_unit=add_unit)

    def label_add_sval(self, stypes=None, add_stype=False, add_unit=True):
        for m in self.measurements:
            m.label_add_stype(stypes=stypes, add_stype=add_stype, add_unit=add_unit)

    def remove_label(self):
        for m in self.measurements:
            m.label = ''
        for m in self.mean_measurements:
            m.label = ''

    ####################################################################################################################
    ''' XML io'''

    @property
    def etree(self):
        """
        Returns the content of the sample as an xml.etree.ElementTree object which can be used to construct xml
        representation

        Returns
        -------
             etree: xml.etree.ElementTree
        """

        sample_node = etree.Element(type(self).SAMPLE, attrib={type(self).NAME: str(self.name)})

        # add list of samplegroups
        for sg in self._samplegroups:
            etree.SubElement(sample_node, type(self).SAMPLEGROUP).text = sg

        # add list of measurements
        for m in self.measurements:
            sample_node.append(m.etree)

        # add list of mean measurements
        for m in self.mean_measurements:
            sample_node.append(m.etree)

        return sample_node

    @classmethod
    def from_etree(cls, et_element):
        if et_element.tag != cls.SAMPLE:
            log.error('XML Import: Need <{}> node to construct object.'.format(cls.SAMPLE))
            return None

        # create sample
        name = et_element.attrib['name']
        s = cls(name=name)

        log.info("XML Import: Reading sample <{}>".format(name))

        # add sample to samplegroups
        for sg in et_element.findall(cls.SAMPLEGROUP):
            s.add_to_samplegroup(sg.text)

        # add measurements
        for m in et_element.findall(RockPy3.core.measurement.Measurement.MEASUREMENT):
            mobj = RockPy3.core.measurement.Measurement.from_etree(m, s)

            s.add_measurement(mobj=mobj)

        # return sample
        return s


class MeanSample(Sample):
    MeanSample = 0

    def __init__(self, name, coord=None,
                 results_from_mean_data=False,
                 study=None,
                 **options):
        """
        Parameters
        ----------
           name: str
              name of the sample.
           coord: str
              coordinate system
              can be 'core', 'geo' or 'bed'
          results_from_mean_data: bool
            if True: the results will be calculated from the mean data
            if False: the results will be the mean of the results for each measurement
        """
        self.name = name
        # coordinate system
        self._coord = coord

        RockPy3.logger.info('CREATING\t MEAN-sample << %s >>' % self.name)

        self.base_measurements = []
        self.measurements = []
        self.results = None
        self.results_from_mean_data = results_from_mean_data

        ''' is sample is a mean sample from sample_goup ect... '''
        self.mean_measurements = []
        self._mean_results = None

        # dictionaries
        self._mdict = self._create_mdict()
        self._mean_mdict = self._create_mdict()
        self._rdict = self._create_mdict()

        self.index = MeanSample.snum
        self._samplegroups = []
        self.study = RockPy3.Study
        self.idx = MeanSample.snum
        MeanSample.snum += 1

    def __repr__(self):
        return '<< RockPy.MeanSample.{} >>'.format(self.name)


if __name__ == '__main__':
    from pprint import pprint

    S = RockPy3.Study
    s = S.add_sample('1440')
    s.add_measurement(fpyth = '/Users/mike/Dropbox/experimental_data/FeNiX/FeNi20K/FeNi_FeNi20-Ka2160\'-G03_HYS_VSM#50,3[mg]_[]_[]##STD030.001', mtype='hys', ftype='vsm')
    # s = S.add_sample(name='test')
    # for n in range(10):
    #     m = s.add_simulation(mtype='hysteresis')
    # pprint(m.standards_result())
    # print(s.get_result(result='ms', test=7, saturation_percent=80))
