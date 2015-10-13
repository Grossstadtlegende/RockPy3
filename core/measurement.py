import core.utils

__author__ = 'volk'

import logging
import inspect
import itertools
import xml.etree.ElementTree as etree

from copy import deepcopy
import numpy as np

import os, pwd
import os.path

import decorator

import RockPy3.core.io

try:
    from pylatex import Document, Section, Subsection, Tabular, Description, Math, TikZ, Axis, Plot, Figure, Package
    from pylatex.utils import italic, escape_latex
    import RockPy3.utils.latex

    pylatex_available = True

except ImportError:
    pylatex_available = False
    logger = logging.getLogger('RockPy')
    logger.warning('Please install module pylatex for latex output.')


class Measurement(object):
    """
    """

    logger = logging.getLogger('RockPy3.MEASUREMENT')
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 100
    possible_plt_props = ['agg_filter', 'alpha', 'animated', 'antialiased', 'axes', 'clip_box', 'clip_on', 'clip_path',
                          'color', 'contains', 'dash_capstyle', 'dash_joinstyle', 'dashes', 'drawstyle', 'figure',
                          'fillstyle', 'gid', 'label', 'linestyle', 'linewidth', 'lod', 'marker', 'markeredgecolor',
                          'markeredgewidth', 'markerfacecolor', 'markerfacecoloralt', 'markersize', 'markevery',
                          'path_effects', 'picker', 'pickradius', 'rasterized', 'sketch_params', 'snap',
                          'solid_capstyle', 'solid_joinstyle', 'transform', 'url', 'visible', 'xdata', 'ydata',
                          'zorder']

    ####################################################################################################################
    ''' calculation parameter methods '''

    @classmethod
    def mtype_calculation_parameter_list(cls):
        """
        searches through all implemented calculation methods and collects the parameters used in the method.

        Returns
        -------
            dictionary
             key == mtype
             value == a sorted list of unique parameters
        """
        gpcp = {}
        for mname, measurement in RockPy3.implemented_measurements.items():
            gpcp.setdefault(mname, sorted(
                list(set([i for j in measurement.possible_calculation_parameter().values() for i in j]))))
        return gpcp

    @classmethod
    def mtype_calculation_parameter(cls):
        """
        searches through all implemented calculation methods and collects the parameters used in the method.

        Returns
        -------
            a sorted list of unique parameters
        """
        gpcp = {}
        for mname, measurement in RockPy3.implemented_measurements.items():
            gpcp.setdefault(mname, measurement.possible_calculation_parameter())
        return gpcp

    @classmethod
    def method_calculation_parameter_list(cls):
        """
        searches through all implemented calculation methods and collects the parameters used in the method.

        Returns
        -------
            a dictionary with:
            key: calculation_method name
            value: list of unique parameters
        """
        gpcp = {}
        for mname, measurement in RockPy3.implemented_measurements.items():
            for method, parameters in measurement.possible_calculation_parameter().items():
                # method_name = ''.join(method.split('_')[1:])  # only method without the _calculate
                gpcp.setdefault(method, parameters)
        return gpcp

    @classmethod
    def global_calculation_parameter(cls):
        """
        searches through all implemented calculation methods and collects the parameters used in the method.

        Returns
        -------
            a sorted list of unique parameters
        """
        gpcp = []
        for mname, measurement in RockPy3.implemented_measurements.items():
            for method, parameters in measurement.possible_calculation_parameter().items():
                gpcp.extend(parameters)
        return sorted(list(set(gpcp)))

    @classmethod
    def possible_calculation_parameter(cls):
        """
        All possible calculation methods of a measurement. It first creates a dictionary of all calculate_methods and
        their signature.
        Then it looks through all result_ methods and checks if any is not added yet and needs a recipe
        Note
        ----
            This adds a pseudo_method to the dictionary for each result_method with a recipe.
            Therefor this dict can not be used to check for methods!
        """

        cparams = {i: set(arg for arg, value in inspect.signature(getattr(cls, 'calculate_' + i)).parameters.items()
                          if not arg in ['self', 'non_method_parameters'])
                    for i in cls.calculate_methods()}

        # methods with recipe need 'recipe' to be added to the calculation_parameters
        for res in cls.result_methods():
            if res not in cparams:
                methods = cls.get_calculate_methods(res)
                if cls.result_category(res) == 'indirect':
                    cparams.setdefault(res, set())
                    cparams[res].update(set(param for recipe in methods for param in cparams[recipe]))
                if 'recipe' in cls.result_category(res):
                    cparams.setdefault(res, set())
                    cparams[res].update(set(param for recipe in methods for param in cparams[recipe]))
                    cparams[res].update(['recipe'])

        return cparams

    """
    ####################################################################################################################
    RESULT/CALCULATE METHOD RELATED
    """

    @classmethod
    def has_recipe(cls, res):
        """
        checks if result has parameter recipe in signature. if it does this means that the result_method has
        different
        """
        if 'recipe' in inspect.signature(getattr(cls, 'result_' + res)).parameters:
            return True
        else:
            return False

    @classmethod
    def has_calculation_method(cls, res):
        """
        checks if result has parameter calculation_method in argspec. if it does this means that the result_method has
        different

        Returns
        -------
            bool
        """
        if 'calculation_method' in inspect.signature(getattr(cls, 'result_' + res)).parameters:
            return True
        else:
            return False

    @classmethod
    def result_category(cls, res):
        """
        Takes a result_method name as argument and checks which category it belongs to.

            1. direct methods:
                :code:`result_a` calls :code:`calculate_a`

            2. direct recipe methods:
                :code:`result_b` calls :code:`calculate_b_RECIPE1` or :code:`calculate_b_RECIPE2`

            3. indirect methods:
                :code:`result_d` calls :code:`calculate_c` via calculation_method parameter

            4. indirect recipe methods:
                :code:`result_f` calls :code:`calculate_e_RECIPE1` or :code:`calculate_e_RECIPE2` via
                calculation_method parameter

        Parameter
        ---------

            result: str
                the name of the result_method without the :code:`'result_'`

        Returns
        -------

            category: str
                'direct' if case 1
                'direct_recipe' if case 1
                'indirect' if case 1
                'indirect_resipe' if case 1
        """
        res = res.replace('result_', '')

        if cls.has_recipe(res) and cls.has_calculation_method(res):
            return 'indirect_recipe'
        elif cls.has_calculation_method(res) and not cls.has_recipe(res):
            return 'indirect'
        elif cls.has_recipe(res) and not cls.has_calculation_method(res):
            return 'direct_recipe'
        elif res in cls.calculate_methods():
            return 'direct'

    @classmethod
    def get_calculate_methods(cls, res):
        """
        takes a result_method name as input and returns list of matching calculate_methods
        """
        # print self
        argspec = dict(zip(inspect.getargspec(getattr(cls, 'result_' + res))[0][1:],
                           inspect.getargspec(getattr(cls, 'result_' + res))[3]))
        # print result
        calculation_method = argspec.get('calculation_method')
        if cls.result_category(res) == 'direct':
            return [res]
        elif cls.result_category(res) == 'indirect':
            return [calculation_method]
        elif cls.result_category(res) == 'direct_recipe':
            return [i for i in cls.calculate_methods()
                    if i.split('_')[-1].isupper
                    if ''.join(i.split('_')[:-1]) == res]
        elif cls.result_category(res) == 'indirect_recipe':
            return [i for i in cls.calculate_methods()
                    if i.split('_')[-1].isupper
                    if ''.join(i.split('_')[:-1]) == calculation_method]

    @classmethod
    def from_simulation(cls, **parameter):
        """
        pseudo abstract method that should be overridden in subclasses to return a simulated measurement
        based on given parameters
        """
        return None

    @classmethod
    def measurement_result(cls, **parameter):
        """
        pseudo abstract method that should be overridden in subclasses to return a measurement created from results
        """
        return None

    @classmethod
    def implemented_ftypes(cls):
        # setting implemented machines
        # looking for all subclasses of RockPy3.io.base.Machine
        # generating a dictionary of implemented machines : {implemented out_* method : machine_class}
        implemented_ftypes = {cl.__name__.lower(): cl for cl in RockPy3.core.io.ftype.__subclasses__()}
        return implemented_ftypes

    @classmethod
    def subclasses(cls):
        """
        Returns a list of the implemented_visuals names
        """
        return [i.__name__.lower() for i in cls.inheritors()]

    @classmethod
    def inheritors(cls):
        """
        Method that gets all children and childrens-children ... from a class

        Returns
        -------
           list
        """
        subclasses = set()
        work = [cls]
        while work:
            parent = work.pop()
            for child in parent.__subclasses__():
                if child not in subclasses:
                    subclasses.add(child)
                    work.append(child)
        return subclasses

    @classmethod
    def implemented_measurements(cls):
        """
        method that dynamically creates a dictionary with "obj.name : obj" entries

        Returns
        -------
           dict
        """
        return {i.__name__.lower(): i for i in Measurement.inheritors()}

    @classmethod
    def measurement_formatters(cls):
        # measurement formatters are important!
        # if they are not inside the measurement class, the measurement has not been implemented for this machine.
        # the following machine formatters:
        # 1. looks through all implemented measurements
        # 2. for each measurement stores the machine and the applicable readin class in a dictionary
        measurement_formatters = \
            {cl.__name__.lower(): {
                '_'.join(i.split('_')[1:]).lower():
                    Measurement.implemented_ftypes()['_'.join(i.split('_')[1:]).lower()] for i
                in dir(cl) if i.startswith('format_')} for cl in Measurement.inheritors()}
        return measurement_formatters

    ####################################################################################################################
    # builtin methods
    @classmethod
    def result_methods(cls):
        """
        Searches through all :code:`result_*` methods and creates a dictionary with:

            result_name : result_method

        where result_name is the name without the result_ prefix
        """
        result_methods = {i[7:]: getattr(cls, i) for i in dir(cls) if i.startswith('result_')
                          if not i.endswith('generic')
                          if not i.endswith('methods')
                          if not i.endswith('category')
                          }
        return result_methods

    @classmethod
    def calculate_methods(cls):
        # dynamically generating the calculation and standard parameters for each calculation method.
        # This just sets the values to non, the values have to be specified in the class itself
        calculate_methods = {i.replace('calculate_', ''): getattr(cls, i) for i in dir(cls)
                             if i.startswith('calculate_')
                             if not i.endswith('generic')
                             if not i.endswith('result')
                             if not i.endswith('methods')
                             }
        return calculate_methods

    @classmethod
    def correct_methods(cls):
        # dynamically generating the calculation and standard parameters for each calculation method.
        # This just sets the values to non, the values have to be specified in the class itself
        methods = {i.replace('correct_', ''): getattr(cls, i) for i in dir(cls)
                   if i.startswith('correct_')
                   }
        return methods

    ####################################################################################################################

    @classmethod
    def get_subclass_name(cls):
        return cls.__name__

    @property
    def study(self):
        """
        find study to which the measurement belongs

        :return:
        """
        return self.sobj.study

    ####################################################################################################################
    # plotting / legend properties
    @property
    def plt_props(self):
        std = {'label': ''}
        if '_plt_props' not in self.__dict__:
            self._plt_props = std
        return self._plt_props

    def set_plt_prop(self, prop, value):
        """
        sets the plt_props for the measurement.

        raises
        ------
            KeyError if the plt_prop not in the matplotlib.lines.Line2D
        """
        if prop not in Measurement.possible_plt_props:
            raise KeyError
        self._plt_props.setdefault(prop, value)
        # self._plt_props[prop] = value

    ####################################################################################################################

    @property
    def coord(self):
        """
        find coordinate system of sample
        """
        print("returning" + str(self.sobj.coord))
        return self.sobj.coord

    def __init__(self, sobj,
                 mtype=None, fpath=None, ftype=None, mdata=None,
                 series=None,
                 **options):
        """
        Constructor of the measurement class. This is called by all measurements using super.

        Several checks have to be done:
            1. is the measurement implemented:
                this is checked by looking if the measurement is in the Measurement.measurement_formatters dictionary

            2. if mdata is given, we can directly create the measurement
            3. if the file format (ftype) is implemented
                The ftype has to be given. This is how RockPy can format data from different sources into the same
                format, so it can be analyzed the same way.

        Parameters
        ----------
            @type sobj: RockPy3.Sample
                the sample object the measurement belongs to. The constructor is usually called from the
                Sample.add_measurement method
            @type mtype: str
                MANDATORY: measurement type to be imported. Implemented measurements can be seen when calling
                >>> print Measurement.measurement_formatters()
           mfile:
           machine:
           mdata: when mdata is set, this will be directly used as measurement data without formatting from file
           color: color used for plotting if specified
           options:
        :return:
        """

        self.logger = logging.getLogger('RockPy3.MEASURMENT.' + self.get_subclass_name())

        # the data that is used for calculations and corrections
        self._data = {}

        # _raw_data is a backup deepcopy of _data it will be used to reset the _data if reset_data() is called
        self._raw_data = {}

        # coordinate system that is currently used in data; _raw_data is always assumed to be in core coordiantes
        self._actual_data_coord = 'core'

        self.is_mean = False  # flag for mean measurements
        self.base_measurements = []  # list with all measurements used to generate the mean

        if ftype is not None:
            ftype = ftype.lower()  # for consistency in code

        if mtype is not None:
            mtype = mtype.lower()  # for consistency in code

        ''' initialize parameters '''
        self.ftype_data = None  # returned data from io.ftype()

        ''' initial state '''
        self.is_initial_state = False
        self.is_ftype_data = None  # returned data from io.ftype()
        self.initial_state = None

        ''' calibration, correction and holder'''
        self.calibration = None
        self.holder = None
        self._correction = []


        self.__initialize()

        if mtype in Measurement.measurement_formatters():
            self.logger.debug('MTYPE << %s >> implemented' % mtype)
            self.mtype = mtype  # set mtype

            if mdata is not None:  # we have mdata -> ignore fpath and just use that data directly
                self.logger.debug('mdata passed -> using as measurement data without formatting')
                self.sobj = sobj
                self._data = mdata
            # ftype == combined for special measurements that need two measurements e.g. REM'
            elif ftype in Measurement.measurement_formatters()[mtype] or ftype in ['combined', 'result']:
                self.logger.debug('ftype << %s >> implemented' % ftype)
                self.ftype = ftype  # set machine
                self.sobj = sobj  # set sobj
                if not fpath:
                    if mtype in RockPy3.Packages.Generic.Measurements.parameters.Parameter.subclasses():
                        self.logger.info(
                            'NO fpath passed -> but measurement Parameter subclass, likely no fpath needed')
                    else:
                        self.logger.warning('NO fpath passed -> no raw_data will be generated')
                else:
                    self.fpath = fpath
                    self.import_data()
            else:
                self.logger.error('UNKNOWN ftype: << %s >>' % ftype)
                self.logger.error(
                    'most likely cause is the \"format_%s\" method is missing in the measurement << %s >>' % (
                        ftype, mtype))
                return
        else:
            self.logger.error('UNKNOWN\t MTYPE: << %s >>' % mtype)
            return

        # SAMPLE IMPORT DONE
        # now we can start the formatting

        # add series if provided
        ''' series '''
        self._series = []
        self.add_series(series=series)

        # dynamic data formatting
        # checks if format_'ftype' exists. If exists it formats self.raw_data according to format_'ftype'
        if ftype == 'combined':
            pass
        elif ftype == 'simulation':
            pass
        elif ftype == 'result':
            pass
        elif callable(getattr(self, 'format_' + ftype)):
            self.logger.debug('FORMATTING raw data from << %s >>' % ftype)
            getattr(self, 'format_' + ftype)()
        else:
            self.logger.error(
                'FORMATTING raw data from << %s >> not possible, probably not implemented, yet.' % ftype)
            return

    def __repr__(self):
        if self.is_mean:
            add = 'mean_'
        else:
            add = ''
        return '<<RockPy3.{}.{}{}{} at {}>>'.format(self.sobj.name, add, self.mtype,
                                                    self.stype_sval_tuples,
                                                    hex(id(self)))

    def __getstate__(self):
        """
        returned dict will be pickled
        :return:
        """
        pickle_me = {k: v for k, v in self.__dict__.items() if k in
                     (
                         'mtype', 'ftype', 'fpath',
                         # plotting related
                         '_plt_props',
                         'calculation_parameters',
                         # data related
                         'has_data',  # 'ftype_data', #removed ftype_data because it is not needed post import
                         '_raw_data', '_data', 'temperature',
                         'initial_state', 'is_initial_state', 'is_mean',
                         # sample related
                         'sobj',
                         '_series_opt', '_series',
                         'suffix',
                         'calibration', 'holder', 'correction',
                     )
                     }
        return pickle_me

    def __setstate__(self, d):
        """
        d is unpickled data
           d:
        :return:
        """
        self.__dict__.update(d)
        self.__initialize()

    def get_calc_method(self, method):
        result_name, recipe = get_result_recipe_name(method)
        self.calculation_recipes.setdefault(result_name, dict()).setdefault(recipe, method)

    def __initialize(self):
        """
        Initialize function is called inside the __init__ function, it is also called when the object is reconstructed
        with pickle.

        :return:
        """

        self.results = RockPy3.Data(
            column_names=self.result_methods(),
            data=[np.nan for i in self.result_methods()])  # dynamic entry creation for all available result methods

        # dynamically generating the calculation and standard parameters for each calculation method.
        # This just sets the values to non, the values have to be specified in the class itself
        self.calculation_methods = {i: getattr(self, i) for i in dir(self)
                                    if i.startswith('calculate_')
                                    if not i.endswith('generic')
                                    if not i.endswith('result')
                                    }

        self.calculation_parameter = {}

        self.is_normalized = False  # normalized flag for visuals, so its not normalized twice
        self.norm = None  # the actual parameters

        self._info_dict = self.__create_info_dict()

    @property
    def stype_sval_tuples(self):
        if self.get_series():
            return [(s.stype, s.value) for s in self.series]
        else:
            return []

    @property
    def m_idx(self):
        return self.sobj.measurements.index(self)  # todo change could be problematic if changing the sobj

    @property
    def fname(self):
        """
        Returns only filename from self.file

        Returns
        -------
           str: filename from full path
        """
        return os.path.split(self.fpath)[-1]

    def import_data(self, rtn_raw_data=None, **options):
        """
        Importing the data from mfile and machine
           rtn_raw_data:
           options:
        :return:
        """

        self.logger.info('IMPORTING << %s , %s >> data' % (self.ftype, self.mtype))

        ftype = options.get('ftype', self.ftype)
        mtype = options.get('mtype', self.mtype)
        fpath = options.get('fpath', self.fpath)
        raw_data = self.measurement_formatters()[mtype][ftype](fpath, self.sobj.name)
        if raw_data is None:
            self.logger.error('IMPORTING\t did not transfer data - CHECK sample name and data file')
            return
        else:
            if rtn_raw_data:
                self.logger.info('RETURNING raw_data for << %s , %s >> data' % (ftype, mtype))
                return raw_data
            else:
                self.ftype_data = raw_data

    def set_initial_state(self,
                          mtype, fpath, ftype,  # standard
                          **options):
        """
        creates a new measurement (ISM) as initial state of base measurement (BSM).
        It dynamically calls the measurement _init_ function and assigns the created measurement to the
        self.initial_state value. It also sets a flag for the ISM to check if a measurement is a MIS.

        Parameters
        ----------
           mtype: str
              measurement type
           mfile: str
              measurement data file
           machine: str
              measurement machine
           options:
        """
        mtype = mtype.lower()
        ftype = ftype.lower()

        self.logger.info('CREATING << %s >> initial state measurement << %s >> data' % (mtype, self.mtype))
        # implemented = {i.__name__.lower(): i for i in Measurement.implemented_visuals()}

        # can only be created if the measurement is actually implemented
        if mtype in RockPy3.implemented_measurements():
            self.initial_state = RockPy3.implemented_measurements()[mtype](self.sobj, mtype, fpath, ftype)
            # self.initial_state.sobj = self.sobj
            # print self.initial_state.sobj
            self.initial_state.is_initial_state = True
            return self.initial_state
        else:
            self.logger.error('UNABLE to find measurement << %s >>' % mtype)

    @property
    def has_initial_state(self):
        """
        checks if there is an initial state
        """
        return True if self.initial_state else False

    ####################################################################################################################
    ''' INFO DICTIONARY '''

    @property
    def info_dict(self):
        if not hasattr(self, '_info_dict'):
            self._info_dict = self.__create_info_dict()
        if not all(i in self._info_dict['series'] for i in self.series):
            self._populate_info_dict()
        return self._info_dict

    def __create_info_dict(self):
        """
        creates all info dictionaries

        Returns
        -------
           dict
              Dictionary with a permutation of ,type, stype and sval.
        """
        d = ['stype', 'sval']
        keys = ['_'.join(i) for n in range(3) for i in itertools.permutations(d, n) if not len(i) == 0]
        out = {i: {} for i in keys}
        out.update({'series': []})
        return out

    def _populate_info_dict(self):
        """
        Removes old info_dict and then Re-calculates the info_dictionary for the measurement
        """
        self._info_dict = self.__create_info_dict()
        map(self.add_s2_info_dict, self.series)

    def add_s2_info_dict(self, series):
        """
        adds a measurement to the info dictionary.

        Parameters
        ----------
           series: RockPy3.Series
              Series to be added to the info_dictionary
        """

        if not series in self._info_dict['series']:
            self._info_dict['stype'].setdefault(series.stype, []).append(self)
            self._info_dict['sval'].setdefault(series.value, []).append(self)

            self._info_dict['sval_stype'].setdefault(series.value, {})
            self._info_dict['sval_stype'][series.value].setdefault(series.stype, []).append(self)
            self._info_dict['stype_sval'].setdefault(series.stype, {})
            self._info_dict['stype_sval'][series.stype].setdefault(series.value, []).append(self)

            self._info_dict['series'].append(series)

    def remove_s_from_info_dict(self, series):
        """
        removes a measurement to the info dictionary.

        Parameters
        ----------
           series: RockPy3.Series
              Series to be removed from the info_dictionary
        """
        if series in self._info_dict['series']:
            self.series.remove(series)
            self._info_dict['stype'].setdefault(series.stype, []).remove(self)
            self._info_dict['sval'].setdefault(series.value, []).remove(self)
            self._info_dict['sval_stype'][series.value].setdefault(series.stype, []).remove(self)
            self._info_dict['stype_sval'][series.stype].setdefault(series.value, []).remove(self)
            self._info_dict['series'].remove(series)
        self._info_dict_cleanup()

    def remove_all_series(self):
        """
        removes all series from the measurement
        """
        for s in self.series:
            self.series.remove(s)
            self.remove_s_from_info_dict(s)

    def _info_dict_cleanup(self):
        """
        recursively removes all empty lists from dictionary
        :param empties_list:
        :return:
        """

        mdict = getattr(self, 'info_dict')

        # cycle through level 0
        for k0, v0 in sorted(mdict.items()):
            if isinstance(v0, dict):
                # cycle through level 1
                for k1, v1 in sorted(v0.items()):
                    if isinstance(v1, dict):
                        for k2, v2 in sorted(v1.items()):
                            if not v2:
                                v1.pop(k2)
                            if not v1:
                                v0.pop(k1)
                            else:
                                if not v1:
                                    v0.pop(k1)
                    else:
                        if not v1:
                            v0.pop(k1)

    @property
    def stypes(self):
        """
        list of all stypes
        """
        return self.info_dict['stype'].keys()

    @property
    def svals(self):
        """
        list of all stypes
        """
        return self.info_dict['sval'].keys()

    @property
    def data(self):
        if self._data == {}:
            self._data = deepcopy(self._raw_data)

        # transform vectorial x,y,z data to new coordinate system when needed
        # self.transform_data_coord(final_coord=self.coord)
        # TODO: fails ->
        # print self.coord

        return self._data

    def transform_data_coord(self, final_coord):
        """
        transform columns x,y,z in data object to final_coord coordinate system
        """
        if self._actual_data_coord != final_coord:
            # check if we have x,y,z columns in data
            # TODO look for orientation measurements
            # extract coreaz, coredip, bedaz and beddip
            # transform x, y, z columns in _data['data']
            # final_xyz = RockPy3.utils.general.coord_transform(initial_xyz, self._actual_data_coord, final_coord)
            self.logger.warning('data needs to be transformed from %s to %s coordinates. NOT IMPLEMENTED YET!' % (
                self._actual_data_coord, final_coord))
            # return self._data
        else:
            self.logger.debug('data is already in %s coordinates.' % final_coord)
            # return self._data

    @property
    def correction(self):
        """
        If attribute _correction does not exist, it is created and returned. After that the attribute always exists.
        This is a way so any _attribute does not have to be created in __init__
        A string of the used calculation method has to be appended for any corrections.
        This way we always know what has been corrected and in what order it has been corrected
        """
        return self.set_get_attr('_correction', value=list())

    def reset_data(self):
        """
        Resets all data back to the original state. Deepcopies _raw_data back to _data and resets correction
        """
        self._data = deepcopy(self._raw_data)
        # create _correction if not exists
        self.set_get_attr('_correction', value=list())
        self._correction = []

    ####################################################################################################################
    ''' DATA RELATED '''

    ''' Calculation and parameters '''

    def calculate_result(self, result, **parameter):
        """
        Helper function to dynamically call a result. Used in Visualize

        Parameters
        ----------
           result:
           parameter:
        """

        if not self.has_result(result):
            self.logger.warning('%s does not have result << %s >>' % self.mtype, result)
            return
        else:
            # todo figure out why logger wrong when called from Visualize
            self.logger = logging.getLogger('RockPy3.MEASURMENT.' + self.mtype + '[%s]' % self.sobj.name)
            self.logger.info('CALCULATING << %s >>' % result)
            out = getattr(self, 'result_' + result)(**parameter)
        return out

    def calc_generic(self, **parameter):
        """
        helper function
        actual calculation of the result

        :return:
        """

        self.results['generic'] = 0

    def calc_result(self, parameter=None, recalc=False, force_method=None):
        '''
        Helper function:
        Calls any calculate_* function, but checks first:

            1. does this calculation method exist
            2. has it been calculated before

               NO : calculate the result

               YES: are given parameters equal to previous calculation parameters

               if YES::

                  NO : calculate result with new parameters
                  YES: return previous result

           parameter: dict
                        dictionary with parameters needed for calculation
           force_caller: not dynamically retrieved caller name.

        :return:
        '''

        caller = '_'.join(inspect.stack()[1][3].split('_')[1:])  # get calling function #todo get rid of inspect

        if not parameter:  # todo streamline the generation of standard parameters
            try:
                parameter = self.standard_parameter[caller]
            except AttributeError:
                parameter = dict(caller={})
            except KeyError:
                parameter = dict(caller={})

        # get the method to be used for calculation. It is either the calling method determined by inspect
        # or the method specified with force_method
        if force_method is not None:
            method = force_method  # method for calculation if any: result_CALLER_method
        else:
            method = caller  # if CALLER = METHOD

        if callable(getattr(self, 'calculate_' + method)):  # check if calculation function exists
            # check for None and replaces it with standard
            parameter = self.compare_parameters(method, parameter, recalc)

            # if results dont exist or force recalc
            if self.results[caller] is None or self.results[caller] == np.nan or recalc:
                # recalc causes a forced racalculation of the result
                if recalc:
                    self.logger.debug('FORCED recalculation of << %s >>' % (method))
                else:
                    self.logger.debug('CANNOT find result << %s >> -> calculating' % (method))
                getattr(self, 'calculate_' + method)(**parameter)  # calling calculation method
            else:
                self.logger.debug('FOUND previous << %s >> parameters' % (method))
                if self.check_parameters(caller, parameter):  # are parameters equal to previous parameters
                    self.logger.debug('RESULT parameters different from previous calculation -> recalculating')
                    getattr(self, 'calculate_' + method)(**parameter)  # recalculating if parameters different
                else:
                    self.logger.debug('RESULT parameters equal to previous calculation')
        else:
            self.logger.error(
                'CALCULATION of << %s >> not possible, probably not implemented, yet.' % method)

    def calc_all(self, recalc=False, **parameter):
        # get possible calculation parameters and put them in a dictionary
        calculation_parameter, kwargs = core.utils.kwargs_to_calculation_parameter(rpobj=self, **parameter)
        for result_method in self.result_methods():
            calc_param = calculation_parameter.get(self.mtype, {})
            calc_param = calc_param.get(result_method, {})
            getattr(self, 'result_' + result_method)(recalc=recalc, **calc_param)
        if kwargs:
            self.logger.warning('--------------------------------------------------')
            self.logger.warning('| %46s |' % 'SOME PARAMETERS COULD NOT BE USED')
            self.logger.warning('--------------------------------------------------')
            for i, v in kwargs.items():
                self.logger.warning('| %22s: %22s |' % (i, v))
            self.logger.warning('--------------------------------------------------')
        if calculation_parameter:
            self.logger.info('--------------------------------------------------')
            self.logger.info('| %46s |' % 'these parameters were used')
            self.logger.info('--------------------------------------------------')
            for i, v in calculation_parameter.items():
                self.logger.info('| %46s |' % i)
                for method, parameter in v.items():
                    self.logger.info('| %22s: %22s |' % (method, parameter))
            self.logger.info('--------------------------------------------------')

        return self.results

    def compare_parameters(self, caller, parameter, recalc):
        """
        checks if given parameter[key] is None and replaces it with standard parameter or calculation_parameter.

        e.g. calculation_generic(A=1, B=2)
             calculation_generic() # will calculate with A=1, B=2
             calculation_generic(A=3) # will calculate with A=3, B=2
             calculation_generic(A=2, recalc=True) # will calculate with A=2 B=standard_parameter['B']

           caller: str
                     name of calling function ('result_generic' should be given as 'generic')
           parameter:
                        Parameters to check
           recalc: Boolean
                     True if forced recalculation, False if not
        :return:
        """
        if not parameter:
            parameter = dict()

        for key, value in parameter.items():
            if value is None:
                if self.calculation_parameter[caller] and not recalc:
                    parameter[key] = self.calculation_parameter[caller][key]
                else:
                    parameter[key] = self.standard_parameter[caller][key]
        return parameter

    def delete_dtype_var_val(self, dtype, var, val):
        """
        deletes step with var = var and val = val

           dtype: the step type to be deleted e.g. th
           var: the variable e.g. temperature
           val: the value of that step e.g. 500

        example: measurement.delete_step(step='th', var='temp', val=500) will delete the th step where the temperature is 500
        """
        idx = self._get_idx_dtype_var_val(dtype=dtype, var=var, val=val)
        self.data[dtype] = self.data[dtype].filter_idx(idx, invert=True)
        return self

    def check_parameters(self, caller, parameter):
        """
        Checks if previous calculation used the same parameters, if yes returns the previous calculation
        if no calculates with new parameters

        Parameters
        ----------
           caller: str
               name of calling function ('result_generic' should be given as 'generic')
           parameter:
        Returns
        -------
           bool
              returns true is parameters are not the same
        """
        if self.calculation_parameter[caller]:
            # parameter for new calculation
            a = []
            for key in self.calculation_parameter[caller]:
                if key in parameter:
                    a.append(parameter[key])
                else:
                    a.append(self.calculation_parameter[caller][key])
                    # a = [parameter[i] for i in self.calculation_parameter[caller]]
            # get parameter values used for calculation
            b = [self.calculation_parameter[caller][i] for i in self.calculation_parameter[caller]]
            if a != b:
                return True
            else:
                return False
        else:
            return True

    def has_result(self, result):
        """
        Checks if the measurement contains a certain result

        Parameters
        ----------
           result: str
              the result that should be found e.g. result='ms' would give True for 'hys' and 'backfield'
        Returns
        -------
           out: bool
              True if it has result, False if not
        """
        if result in self.result_methods():
            return True
        else:
            return False

    """
    ####################################################################################################################

    SERIES related
    """

    @property
    def series(self):
        if self._series:
            return self._series
        else:
            series = RockPy3.Series(stype='none', value=np.nan, unit='')
            return [series]

    def get_series(self, stype=None, sval=None, series=None):
        """
        searches for given stypes and svals in self.series and returns them

        Parameters
        ----------
            series: list(tuple)
                list of tuples to avoid problems wit separate series and same sval
            stypes: list, str
                stype or stypes to be looked up
            svals: float
                sval or svals to be looked up

        Returns
        -------
            list
                list of series that fit the parameters
                if no parameters - > all series
                empty if none fit

        Note
        ----
            m = measurement with [<RockPy3.series> pressure, 0.00, [GPa], <RockPy3.series> temperature, 0.00, [C]]
            m.get_series('pressure', 0) -> [<RockPy3.series> pressure, 0.00, [GPa]]
            m.get_series(0) -> [<RockPy3.series> pressure, 0.00, [GPa], <RockPy3.series> temperature, 0.00, [C]]
        """
        out = self.series
        if stype is not None:
            stype = RockPy3.core.utils.to_list(stype)
            stype = [stype.lower() for stype in stype]
            out = [i for i in out if i.stype in stype]
        if sval is not None:
            sval = RockPy3.core.utils.to_list(sval)
            out = [i for i in out if i.value in sval]

        if series:  # todo series

            out = [i for i in out if i.data in sval]

        return out

    def get_sval(self, stype):
        """
        Searches for stype and returns sval
        """
        s = self.get_series(stype=stype)
        return s[0].value if s else None

    def add_series(self, stype=None, sval=None, unit=None, series_obj=None, series=None):
        """
        adds a series to measurement.series, then adds is to the data and results datastructure

        Parameters
        ----------
           stype: str
              series type to be added
           sval: float or int
              series value to be added
           unit: str
              unit to be added. can be None #todo change so it uses Pint
            series_obj: RockPy3.series
                if a previously created object needs to be passed
            series: list(tuples)
                default: None
                Series object gets created for a list of specified series

        Returns
        -------
           [RockPy3.Series] list of RockPy series objects

        Note
        ----
            If the measurement previously had no series, the (none, 0 , none) standard series will be removed first
        """
        # if a series object is provided other wise create series object
        if not any(i for i in [stype, sval, unit, series_obj, series]):
            return
        elif series_obj:
            series = series_obj

        elif series:
            if type(series) == tuple:
                aux = []
                aux.append(series)
                slist = aux
            else:
                slist = series
            series = []
            for stup in slist:
                series.append(RockPy3.Series.from_tuple(series=stup))
        else:
            series = RockPy3.Series(stype=stype, value=sval, unit=unit)

        # remove default series from sobj.mdict if non series exists previously
        if not self._series:
            self.sobj._remove_series_from_mdict(mobj=self, series=self.series[0],
                                                mdict_type='mdict')  # remove default series

        # turn into list to add multiples
        series = RockPy3.core.utils.to_list(series)
        for sobj in series:
            if not any(sobj == s for s in self._series):
                print(sobj)
                self._series.append(sobj)
                self._add_sval_to_data(sobj)
                self._add_sval_to_results(sobj)

            # add the measurement to the mdict of the sobj
            self.sobj._add_series2_mdict(series=sobj, mobj=self)
        return series

    def _add_sval_to_data(self, sobj):
        """
        Adds stype as a column and adds svals to data. Only if stype != none.

        Parameter
        ---------
           sobj: series instance
        """
        if sobj.stype != 'none':
            for dtype in self._raw_data:
                if self._raw_data[dtype]:
                    data = np.ones(len(self.data[dtype]['variable'].v)) * sobj.value
                    if not 'stype ' + sobj.stype in self.data[dtype].column_names:
                        self.data[dtype] = self.data[dtype].append_columns(column_names='stype ' + sobj.stype,
                                                                           data=data)  # , unit=sobj.unit) #todo add units

    def _add_sval_to_results(self, sobj):
        """
        Adds the stype as a column and the value as value to the results. Only if stype != none.

        Parameter
        ---------
           sobj: series instance
        """
        if sobj.stype != 'none':
            # data = np.ones(len(self.results['variable'].v)) * sobj.value
            if not 'stype ' + sobj.stype in self.results.column_names:
                self.results = self.results.append_columns(column_names='stype ' + sobj.stype,
                                                           data=[sobj.value])  # , unit=sobj.unit) #todo add units

    def __sort_list_set(self, values):
        """
        returns a sorted list of non duplicate values
           values:
        :return:
        """
        return sorted(list(set(values)))

    def _get_idx_dtype_var_val(self, dtype, var, val, *args):
        """
        returns the index of the closest value with the variable(var) and the step(step) to the value(val)

        option: inverse:
           returns all indices except this one

        """
        out = [np.argmin(abs(self.data[dtype][var].v - val))]
        return out

    def equal_series(self, other):
        if all(i in other.series for i in self.series):
            return True
        if not self.series and not other.series:
            return True
        else:
            return False

    """
    Normalize functions
    +++++++++++++++++++
    """

    def normalize(self,
                  reference='data', ref_dtype='mag', norm_dtypes='all', vval=None,
                  norm_method='max', norm_factor=None, result=None,
                  normalize_variable=False, dont_normalize=['time', 'temp'],
                  norm_initial_state=True, **options):
        """
        normalizes all available data to reference value, using norm_method

        Parameter
        ---------
            reference: str
                reference state, to which to normalize to e.g. 'NRM'
                also possible to normalize to mass
            ref_dtype: str
                component of the reference, if applicable. standard - 'mag'
            norm_dtypes: list
                default: 'all'
                dtypes to be normalized, if dtype = 'all' all columns will be normalized
            vval: float
                variable value, if reference == value then it will search for the point closest to the vval
            norm_method: str
                how the norm_factor is generated, could be min
            normalize_variable: bool
                if True, variable is also normalized
                default: False
            result: str
                default: None
                normalizes the values in norm_dtypes to the result value.
                e.g. normalize the moment to ms (hysteresis measuremetns)
            dont_normalize: list
                list of dtypes that will not be normalized
                default: None
            norm_initial_state: bool
                if true, initial state values are normalized in the same manner as normal data
                default: True
        """
        # separate the calc from non calc parameters
        calculation_parameter, options = core.utils.separate_calculation_parameter_from_kwargs(rpobj=self,
                                                                                                          **options)

        # getting normalization factor
        if not norm_factor:  # if norm_factor specified
            norm_factor = self._get_norm_factor(reference=reference, rtype=ref_dtype,
                                                vval=vval,
                                                norm_method=norm_method,
                                                result=result,
                                                **calculation_parameter)

        norm_dtypes = _to_tuple(norm_dtypes)  # make sure its a list/tuple
        for dtype, dtype_data in self.data.items():  # cycling through all dtypes in data
            if dtype_data:
                if 'all' in norm_dtypes:  # if all, all non stype data will be normalized
                    norm_dtypes = [i for i in dtype_data.column_names if not 'stype' in i]

                ### DO not normalize:
                # variable
                if not normalize_variable:
                    variable = dtype_data.column_names[dtype_data.column_dict['variable'][0]]
                    norm_dtypes = [i for i in norm_dtypes if not i == variable]

                if dont_normalize:
                    dont_normalize = _to_tuple(dont_normalize)
                    norm_dtypes = [i for i in norm_dtypes if not i in dont_normalize]

                for ntype in norm_dtypes:  # else use norm_dtypes specified
                    try:
                        dtype_data[ntype] = dtype_data[ntype].v / norm_factor
                    except KeyError:
                        self.logger.warning(
                            'CAN\'T normalize << %s, %s >> to %s' % (self.sobj.name, self.mtype, ntype))

                if 'mag' in dtype_data.column_names:
                    try:
                        self.data[dtype]['mag'] = self.data[dtype].magnitude(('x', 'y', 'z'))
                    except KeyError:
                        self.logger.debug('no (x,y,z) data found keeping << mag >>')

        self.logger.debug('NORMALIZING << %s >> with << %.2e >>' % (', '.join(norm_dtypes), norm_factor))

        if self.initial_state and norm_initial_state:
            for dtype, dtype_rpd in self.initial_state.data.items():
                self.initial_state.data[dtype] = dtype_rpd / norm_factor
                if 'mag' in self.initial_state.data[dtype].column_names:
                    self.initial_state.data[dtype]['mag'] = self.initial_state.data[dtype].magnitude(('x', 'y', 'z'))
        return self

    def _get_norm_factor(self, reference, rtype, vval, norm_method, result, **calculation_parameter):
        """
        Calculates the normalization factor from the data according to specified input

        Parameter
        ---------
           reference: str
              the type of data to be referenced. e.g. 'NRM' -> norm_factor will be calculated from self.data['NRM']
              if not given, will return 1
           rtype:
           vval:
           norm_method:

        Returns
        -------
           normalization factor: float
        """
        norm_factor = 1  # inititalize
        if reference and not result:
            if reference == 'nrm' and reference not in self.data and 'data' in self.data:
                reference = 'data'

            if reference in self.data:
                norm_factor = self._norm_method(norm_method, vval, rtype, self.data[reference])

            if reference in ['is', 'initial', 'initial_state']:
                if self.initial_state:
                    norm_factor = self._norm_method(norm_method, vval, rtype, self.initial_state.data['data'])
                if self.is_initial_state:
                    norm_factor = self._norm_method(norm_method, vval, rtype, self.data['data'])

            if reference == 'mass':
                m = self.get_mtype_prior_to(mtype='mass')
                norm_factor = m.data['data']['mass'].v[0]

            if isinstance(reference, float) or isinstance(reference, int):
                norm_factor = float(reference)

        elif result:
            norm_factor = getattr(self, 'result_' + result)(**calculation_parameter)[0]
        else:
            self.logger.warning('NO reference specified, do not know what to normalize to.')
        return norm_factor

    def _norm_method(self, norm_method, vval, rtype, data):
        methods = {'max': max,
                   'min': min,
                   # 'val': self.get_val_from_data,
                   }
        if not vval:
            if not norm_method in methods:
                raise NotImplemented('NORMALIZATION METHOD << %s >>' % norm_method)
                return
            else:
                return methods[norm_method](data[rtype].v)

        if vval:
            idx = np.argmin(abs(data['variable'].v - vval))
            out = data.filter_idx([idx])[rtype].v[0]
            return out

    def get_mtype_prior_to(self, mtype):
        """
        search for last mtype prior to self

        Parameters
        ----------
           mtype: str
              the type of measurement that is supposed to be returned

        Returns
        -------
           RockPy3.Measurement
        """

        measurements = self.sobj.get_measurements(mtypes=mtype)

        if measurements:
            out = [i for i in measurements if i.m_idx <= self.m_idx]
            return out[-1]

        else:
            return None

    def _add_stype_to_results(self):
        """
        adds a column with stype stype.name to the results for each stype in measurement.series
        :return:
        """
        if self._series:
            for t in self.series:
                if t.stype:
                    if t.stype not in self.results.column_names:
                        self.results.append_columns(column_names='stype ' + t.stype,
                                                    data=t.value,
                                                    # unit = t.unit      # todo add units
                                                    )

    def get_series_labels(self, stypes=True, add_stype=True, add_unit=True):
        """
        takes a list of stypes or stypes = True and returns a string with stype_sval_sunit; for each stype

        Parameters
        ----------
            stypes: list / bool
                default: True
                if True all series will be returned
                if a list of strings is provided the ones where a matching stype can be found are appended
            add_stype: bool
                default: True
                if True: stype is first part of label
                if False: stype not in label
            add_unit: bool
                default: True
                if True: unit is last part of label
                if False: unit not in label
        """
        out = []
        if stypes is True:
            stypes = self.stypes

        stypes = _to_tuple(stypes)

        for stype in stypes:
            if self.get_series(stype=stype):
                stype = self.get_series(stype=stype)[0]
                aux = []
                if add_stype:
                    aux.append(stype.stype)
                aux.append(str(np.round(stype.value, 2)))
                if add_unit:
                    aux.append(stype.unit)
                stype_label = ' '.join(aux)
                if not stype_label in out:
                    out.append(stype_label)
            else:
                self.logger.warning('CANT find series << %s >>' % stype)
                self.logger.warning('\tonly one of these are possible:\t%s' % self.stypes)
        return '; '.join(out)

    def has_mtype_stype_sval(self, mtype=None, stype=None, sval=None):
        """
        checks if the measurement type is mtype, if it has a series with stype and the sval and if it is a mean measurement

        Parameters
        ----------

            mtype: str
                checks if it is the correct mtype
            stype: str
                checks if it has series with the correct stype
            sval: float
                checks if it has series with the correct sval
        Note
        ----
            only takes a single str/float for each argument
        """
        if mtype and self.mtype != mtype:
            return False
        if stype is not None or sval is not None:
            if not self.get_series(stype=stype, sval=sval):
                return False
        return True

    """
    CORRECTIONS
    +++++++++++
    """

    def correct_dtype(self, dtype='th', var='variable', val='last', initial_state=True):
        """
        corrects the remaining moment from the last th_step

           dtype:
           var:
           val:
           initial_state: also corrects the initial state if one exists
        """

        try:
            calc_data = self.data[dtype]
        except KeyError:
            self.log.error('REFERENCE << %s >> can not be found ' % (dtype))

        if val == 'last':
            val = calc_data[var].v[-1]
        if val == 'first':
            val = calc_data[var].v[0]

        idx = self._get_idx_dtype_var_val(step=dtype, var=var, val=val)

        correction = self.data[dtype].filter_idx(idx)  # correction step

        for dtype in self.data:
            # calculate correction
            self._data[dtype]['m'] = self._data[dtype]['m'].v - correction['m'].v
            # recalc mag for safety
            self.data[dtype]['mag'] = self.data[dtype].magnitude(('x', 'y', 'z'))
        self.reset__data()

        if self.initial_state and initial_state:
            for dtype in self.initial_state.data:
                self.initial_state.data[dtype]['m'] = self.initial_state.data[dtype]['m'].v - correction['m'].v
                self.initial_state.data[dtype]['mag'] = self.initial_state.data[dtype].magnitude(('x', 'y', 'z'))
        return self

    def set_calibration_measurement(self,
                                    fpath=None,  # file path
                                    mdata=None,
                                    mobj=None,  # for special import of a measurement instance
                                    ):
        """
        creates a new measurement that can be used as a calibration for self. The measurement has to be of the same
        mtype and has to have the same ftype

        Parameters
        ----------
        fpath: str
            the full path and filename where the file is located on the hard disk
        mdata: RockPyData
        mobj: RockPy3.Measurement
        """

        cal = self.sobj.add_measurement(mtype=self.mtype, ftype=self.ftype, fpath=fpath,
                                        mobj=mobj, mdata=mdata,
                                        create_only=True)
        self.calibration = cal

    '''' PLOTTING '''''

    def label_add_sample_name(self):
        """
        adds the corresponding sample_name to the measurement label
        """
        if isinstance(self.sobj, RockPy3.MeanSample):
            mean = 'mean '
        else:
            mean = ''
        self.plt_props['label'] = ' '.join([self.plt_props['label'], mean, self.sobj.name])

    def label_add_sample_group_name(self):
        """
        adds the corresponding sample_name to the measurement label
        """
        self.plt_props['label'] = ' '.join([self.plt_props['label'], self.sample_group_obj.name])

    def label_add_study_name(self):
        """
        adds the corresponding sample_name to the measurement label
        """
        self.plt_props['label'] = ' '.join([self.plt_props['label'], self.study_obj.name])

    def label_add_stype(self, stypes, add_stype=True, add_unit=True):
        """
        adds the corresponding sample_name to the measurement label
        """
        text = self.get_series_labels(stypes=stypes, add_stype=add_stype, add_unit=add_unit)
        self.plt_props['label'] = ' '.join([self.plt_props['label'], text])

    @property
    def plottable(self):
        """
        returns a list of all possible Visuals for this measurement
        :return:
        """
        out = {}
        for name, visual in RockPy3.Visualize.base.Visual.implemented_visuals().items():
            if visual._required == [self.mtype]:
                out.update({visual.__name__.lower(): visual})
        return out

    def show_plots(self):
        for visual in self.plottable:
            self.plottable[visual](self, show=True)

    def set_get_attr(self, attr, value=None):
        """
        checks if attribute exists, if not, creates attribute with value None
           attr:
        :return:
        """
        if not hasattr(self, attr):
            setattr(self, attr, value)
        return getattr(self, attr)

    def series_to_color(self, stype):
        # get all possible svals in the hierarchy #todo implement sg and study
        svals = sorted(self.sobj.mdict['stype_sval'][stype].keys())

        # create colormap from svals
        color_map = RockPy3.Visualize.core.create_heat_color_map(svals)

        # get the index and set the color
        sval = self.get_series(stype=stype)[0].value
        sval_index = svals.index(sval)
        self.color = color_map[sval_index]

    def plt_all(self, **plt_opt):
        fig = RockPy3.Figure()
        calculation_parameter, non_calculation_parameter = core.utils.separate_calculation_parameter_from_kwargs(
            self, **plt_opt)
        for visual in self.plottable:
            fig.add_visual(visual=visual, plt_input=self, **plt_opt)
        fig.show(**non_calculation_parameter)

    ####################################################################################################################
    ''' REPORT '''

    def report(self, author=None, doc=None, filepath=None,
               add_results=True, add_calculation_params=True, add_plots=True,
               generate_pdf=True, clean=False):

        def params_dict_to_text(params):
            """
            turns key-value from calculation params into text
            """
            if not params:
                return 'None'
            out = ''
            for key, value in params.items():
                if not key:
                    text = 'None \\\\'
                else:
                    text = '\\textsc{{{0}}}: {1}'.format(key, value).replace('_', ' ')
                if not out:
                    out = text
                else:
                    out = ';\t'.join([out, text])
            return out

    @property
    def etree(self):
        """
        Returns the content of the measurement as an xml.etree.ElementTree object which can be used to construct xml
        representation

        Returns
        -------
             etree: xml.etree.ElementTree
        """

        measurement_node = etree.Element(tag='measurement', attrib={'mtype': str(self.mtype)})

        return measurement_node

        if not author:
            author = pwd.getpwuid(os.getuid())[0]
        if not filepath:
            filepath = RockPy3.core.file_operations.default_folder
        if not doc:
            doc = Document(title='Measurement Report', author=author, date=RockPy3.utils.general.get_date_str(),
                           maketitle=True)

            for pckg in RockPy3.utils.latex.std_packages:
                doc.packages.append(pckg)

        subsection_name = '{} - {} {}'.format(self.sobj.name, self.mtype,
                                              ','.join(['{} [{}]'.format(i.value, i.unit) for i in self.series]))

        with doc.create(Subsection(subsection_name)):
            ############################################################################################################
            ''' RESULTS TABLE '''

            if add_results:
                doc.append(italic('Results:\\\\'))

                # get all calculated results
                names = sorted([name for name in self.results.column_names if 'stype' not in name])
                names_lists = [names[x:x + 6] for x in xrange(0, len(names), 6)]

                results = [['{res[0]:.3}$\\pm${res[1]:.2}'.format(res=getattr(self, 'result_' + name)())
                            if not np.isnan(getattr(self, 'result_' + name)()[1])
                            else '{res[0]:.3}'.format(res=getattr(self, 'result_' + name)())
                            for name in line]
                           for line in names_lists]

                with doc.create(Tabular('cccccccccc')) as table:
                    for i, row in enumerate(names_lists):
                        table.add_hline()
                        table.add_row([name.replace('result_', '').replace('_', ' ') for name in row])
                        table.add_hline()
                        table.add_row(results[i])
                        table.add_empty_row()
                doc.append('\\\\')

            ############################################################################################################
            ''' CALCULATION PARAMETERS '''

            doc.append(italic('Calculation Parameters:\\\\'))

            if add_calculation_params:
                with doc.create(Description()) as desc:
                    for n in names:
                        n_text = n.replace('_', ' ')
                        if n in self.calculation_parameter:
                            desc.add_item(n_text + ': ', params_dict_to_text(self.calculation_parameter[n]))
                    doc.append('\\\\')

            ############################################################################################################
            ''' CALCULATION PARAMETERS '''
            if add_plots:
                fig = RockPy3.Figure()
                for name, visual in self.plottable.items():
                    figname = os.path.join(filepath, '{}_{}.pdf'.format(self.sobj.name, name))
                    visual = fig.add_visual(name, plt_input=self)
                    visual.title += ' ' + self.sobj.name
                fig.show()

        if generate_pdf:
            doc.generate_pdf(filepath=filepath, clean=clean)


def separate_result_name(res):
    """
    separates the 'result_', 'result_method' and 'RECIPE'
    :param
        res: str
    :return:
    """
    print(res.split('_'))


@decorator.decorator
def result(func, *args, **kwargs):
    """
    Result decorator. This decorator calls the right :code:`calculation_method` from the pool of possible
    methods/recipes of the measurement class.

    it first builds the correct name for the :code:`calculation_method` then calls the method and directly returns the
    results without actually calling the result_method

    Parameters
    ----------
        recipe: str
            the default calculation recipe (e.g linear or nonlinear) for the given result_method
            some results have more than one calculation method implemented these can be used with this
        calculation_method: str
            default: None
            if this is not none, but a string, the calculation of the result will be done using a different
            result method than the specified.
            Some Results (e.g. Thellier.result_sigma) is calculated in a different method (in this case it
            is calculated by Thellier.result_slope) and therefore 'slope' has to be passed
    """
    # compute the parameter dictionary from the functions argspecs
    parameters = core.utils.get_full_argspec(func=func, args=args, kwargs=kwargs)
    # get the measurement object, equivalent to self in class = args[0]
    self = parameters.pop('self')

    calculation_parameters, p = core.utils.separate_calculation_parameter_from_kwargs(self, **parameters)
    print(calculation_parameters, p)
    # calculation method has to be popped from dictionary, otherwise it is stored in calculation parameters
    # when the calculation_method is called
    if 'calculation_method' in parameters:
        calculation_method = parameters.pop('calculation_method')
    else:
        calculation_method = None

    # recipes are the different possible calculation paths
    recipe = parameters.get('recipe', '').upper()  # not pop because needed for calculation_parameter check

    result_name = '_'.join(func.__name__.split('_')[1:])

    # add called from to parameters so output will give correct result otherwise:
    # results that are calculated using calculation_method with show calculation method in output instead of
    # result
    parameters.setdefault('called_from', result_name)
    # get a list of all possible calculation methods
    recipes = [i.split('_')[-1] for i in self.calculation_methods.keys()
               if len(i.split('_')) >= 3
               if calculation_method in i.split('_') or result_name in '_'.join(i.split('_')[1:-1])
               if i.split('_')[-1].isupper()]

    if recipe and recipe not in recipes:
        raise NotImplementedError(
            'RECIPE << {} >> not implemented for {} chose from {}'.format(recipe, result_name, recipes))

    # building the name of the calculation method to be called
    # if a calculation method is given, a different method has to be called.
    # e.g. result_eigenvalue1 needs to call calculate_tensor -> calculation_method needs to be 'tensor
    if calculation_method:
        calc_name = 'calculate_' + calculation_method
    # otherwise we can call the calculation_method with the same name
    else:
        calc_name = 'calculate_' + result_name
    # we need to add any possible methods #todo rename from method
    if recipe:
        calc_name += '_' + str(recipe)

    if calc_name not in self.calculation_methods:
        raise NotImplementedError('CALCULATION METHOD << %s >> not implemented' % calc_name)

    # call calculation method, all args per passed
    self.calculation_methods[calc_name](**parameters)
    return self.results[result_name].v[0], self.results[result_name].e[0]


@decorator.decorator
def calculate(func, *args, **kwargs):
    """
    Compares the parameters of calculated result with parameter used for new calculation.

    There are three possible, where the result has to be calculated

    #. the result has not been calculated before:
        this can be seen in the self.calculation_parameters dictionary because it will not have a key for the
        calculation_method_name
    #. the result is forced to be recalculated:
        parameters also has an entry for recalc, which is :code:`True`
    #. at least one parameter differs from the calculation_parameters

    If any of the above is true, the method then updates the calcualtion_parameters with the new parameters and
    returns True

    Parameters
    ----------
        calculation_method_name: str
            the name of the result that should be calculated. Used for looking up the previous calculation_parameters
            in the :code:`self.calcluation_parameter` dictionary.
        method_name: str
            the method that is supposed to be used to compute the result. this comes directly from
            :code:`self.calculate_result_name_MethodName`
        parameters: dict
            the calculation parameters used for the new calculation, these will be returned if the calculation should
            be done
        recalc: bool
            True if the calculation is forced

    Returns
    -------
        False
            if paramters are equal -> no new calculation necessary
        True
            if parameters are not equal -> new calculation with new parameters
    """
    parameters = {arg: args[i] for i, arg in enumerate(decorator.getfullargspec(func).args)}

    calculation_method_name, recipe_name = get_result_recipe_name(func_name=func.__name__)

    if 'called_from' in kwargs:
        called_from = kwargs.pop('called_from')
    else:
        called_from = calculation_method_name

    # only add recipe to name if recipe is given, decluttering calculation_parameter dict
    if recipe_name:
        parameters.update(dict(recipe=recipe_name))

    parameters.pop('check',
                   None)  # pop check from parameters in case check is passed for calculations, None ignores error

    # get recalc value from locals dictionary passed from method
    recalc = kwargs.pop('recalc', False)

    # get the measurement object, equivalent to self in class
    self = parameters.pop('self')

    self.logger.debug('CHECKING if << %s >> needs to be called' % func.__name__)

    do_calc = False  # flag, that shows if result should be returned or the result
    # recalc forces a new calculation
    if recalc:
        self.calculation_parameter.setdefault(calculation_method_name, parameters)
        self.logger.debug('FORCED recalculation of << %s >>' % called_from)
        self.logger.debug('\t with: %s' % parameters)
        do_calc = True
    # if result has not been calculated so far,
    # self.calculation_parameter[calculation_method_name] is an empty dictionary
    elif calculation_method_name not in self.calculation_parameter:
        self.calculation_parameter.setdefault(calculation_method_name, parameters)
        self.logger.debug('RESULT << %s >> not calculated yet' % called_from)
        self.logger.debug('\tcalculation with: %s' % parameters)
        do_calc = True
    elif 'recipe' in parameters \
            and not parameters['recipe'] == self.calculation_parameter[calculation_method_name]['recipe']:
        self.logger.debug('RESULT << %s >> calculated with different recipe' % calculation_method_name)
        do_calc = True
    # # if only one of the parameters is different result has to be recalculated
    elif any(vold != parameters[key] for key, vold in self.calculation_parameter[calculation_method_name].items()):
        self.logger.debug('RESULT << %s >> parameters have changed' % called_from)
        self.logger.debug('\told parameters: %s' % self.calculation_parameter[calculation_method_name])
        self.logger.debug('\tnew parameters: %s' % parameters)
        self.calculation_parameter[calculation_method_name].update(parameters)
        do_calc = True
    if do_calc:
        self.logger.info('CALCULATING << %s >>' % called_from)
        return func(*args, **kwargs)
    else:
        self.logger.debug('RESULT << %s >> already calculated' % called_from)
        self.logger.debug('\twith parameters: %s' % self.calculation_parameter[calculation_method_name])


@decorator.decorator
def correction(func, *args, **kwargs):
    """
    automatically adds the called correction_function to self._correct
    """
    self = args[0]
    self.correction.append(func.__name__)
    return func(*args, **kwargs)


def get_result_recipe_name(func_name):
    """
    Takes the name of a result or a calculation method and extracts the result name and the method ame from it

    Returns
    -------
        result_name: str
        method_name: str
    """

    full_name = func_name.replace('calculate_', '').replace('result_', '')
    split = full_name.split('_')

    if split[-1].isupper():
        recipe = split[-1].lower()
        result_name = '_'.join(split[:-1])
    else:
        recipe = 'none'
        result_name = full_name

    return result_name, recipe
