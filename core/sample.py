import logging

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

        self._samplegroups = core.utils.to_list(samplegroup)

        # coordinate system
        self._coord = coord

        RockPy3.logger.info('CREATING\t new sample << %s >>' % self.name)

        self.raw_measurements = []
        self.measurements = []
        self.results = None

        ''' is sample is a mean sample from samplegroup ect... '''
        # self.is_mean = False  # if a calculated mean_sample #todo needed?
        # self.mean_measurements = []
        # self._mean_results = None

        # # dictionaries
        # self._mdict = self._create_mdict()
        # self._mean_mdict = self._create_mdict()
        # self._rdict = self._create_mdict()

        # if mass is not None:
        #     mass = self.add_measurement(mtype='mass', fpath=None, ftype=mass_ftype,
        #                                 value=float(mass), unit=mass_unit)
        # if diameter is not None:
        #     diameter = self.add_measurement(mtype='diameter', fpath=None, ftype=length_ftype,
        #                                     diameter=float(diameter), length_unit=length_unit)
        # if height is not None:
        #     height = self.add_measurement(mtype='height', fpath=None, ftype=length_ftype,
        #                                   height=float(height), length_unit=length_unit)
        #
        # if x_len:
        #     x_len = self.add_measurement(mtype='length', fpath=None, ftype=length_ftype,
        #                                  value=float(x_len), unit=length_unit, direction='x')
        # if y_len:
        #     y_len = self.add_measurement(mtype='length', fpath=None, ftype=length_ftype,
        #                                  value=float(y_len), unit=length_unit, direction='y')
        # if z_len:
        #     z_len = self.add_measurement(mtype='length', fpath=None, ftype=length_ftype,
        #                                  value=float(z_len), unit=length_unit, direction='z')
        #
        # if height and diameter:
        #     self.add_measurement(mtype='volume', sample_shape=sample_shape, height=height, diameter=diameter)
        # if x_len and y_len and z_len:
        #     self.add_measurement(mtype='volume', sample_shape=sample_shape, x_len=x_len, y_len=y_len, z_len=z_len)

    def __repr__(self):
        return '<< RockPy.Sample.{} >>'.format(self.name)

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
        mobj: RockPy.Measurement object
            if provided, the object is added to self.measurements
        create_only: bool
            if True the measurement is only created and not appended to self.measurements, self.raw_measurements
            or any mdict


        Returns
        -------
            RockPy.measurement object

        :mtypes:

        - mass
        '''
        # abbrev, inv_abbrev = RockPy.core.file_operations.mtype_ftype_abbreviations()
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
                file_info.update(dict(sample_obj=self))
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
                    fname = RockPy.get_fname_from_info(sample_group='SG', sample_name=self.name, mtype=mtype,
                                                       ftype=ftype)
                    self.logger.info('FILE NAME proposition:')
                    self.logger.info('')
                    self.logger.info('                   %s' % fname)
                    self.logger.info('')
                    self.logger.info('Please check for consistency and add sample_group and mass, height, diameter')
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

            file_info = dict(sample_obj=self,
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

            if mtype in Sample.implemented_measurements() or mobj:
                self.logger.info('ADDING\t << measurement >> %s' % mtype)
                if series:
                    self.logger.info(
                        '\t\t WITH series << %s >>' % ('; '.join(', '.join(str(j) for j in i) for i in series)))
                if mobj:
                    measurement = mobj  # todo mobj.sample_obj = self?
                else:
                    # create instance from implemented_measurements dictionary
                    # call constructor of a subclassed measurement
                    measurement = Sample.implemented_measurements()[mtype](**file_info)
                if measurement and not create_only:
                    self.measurements.append(measurement)
                    # append a deepcopy of the measurement to the raw_measurement list so that it is possible to
                    # undo any calculations
                    self.raw_measurements.append(deepcopy(measurement))
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
                self.logger.error('\tIMPLEMENTED: %s' % Sample.implemented_measurements().keys())

    def add_to_samplegroup(self, gname):
        self.logger.debug('ADDING {} to samplegroup {}'.format(self.name, gname))
        self._samplegroups.append(gname)

    def remove_from_samplegroup(self, gname):
        self.logger.debug('REMOVING {} from samplegroup {}'.format(self.name, gname))
        self._samplegroups.remove(gname)
