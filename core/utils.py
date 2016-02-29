import inspect

__author__ = 'mike'
from contextlib import contextmanager
from copy import deepcopy
import RockPy3
import logging
from functools import wraps
import matplotlib.dates
import datetime
import decorator
from pprint import pprint
import numpy as np

def colorscheme(scheme='simple'):
    colors = {'simple': ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 100,
              'pretty': ['k', '#7f0000', '#ed1c24', '#f18c22', '#ffde17', '#add136', '#088743', '#47c3d3', '#21409a',
                         '#96649b', '#ee84b5'] * 100}
    return colors[scheme]


def create_heat_color_map(value_list, reverse=False):
    """
    takes a list of values and creates a list of colors from blue to red (or reversed if reverse = True)

    :param value_list:
    :param reverse:
    :return:
    """
    r = np.linspace(0, 255, len(value_list)).astype('int')
    r = [hex(i)[2:-1].zfill(2) for i in r]
    # r = [i.encode('hex') for i in r]
    b = r[::-1]

    out = ['#%2s' % r[i] + '00' + '%2s' % b[i] for i in range(len(value_list))]
    if reverse:
        out = out[::-1]
    return out


def create_logger(name):
    log = logging.getLogger(name=name)
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(levelname)-10s %(name)-20s %(message)s')
    # formatter = logging.Formatter('%(levelname)-10s %(name)-20s %(message)s')
    fh = logging.FileHandler('RPV3.log')
    # fh.setFormatter(formatter)
    # fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


def convert_time(time):
    return matplotlib.dates.date2num(
        datetime.datetime.strptime(time.replace('.500000', ''), "%Y-%m-%d %H:%M:%S"))


def to_list(oneormoreitems):
    """
    convert argument to tuple of elements
    :param oneormoreitems: single number or string or list of numbers or strings
    :return: list of elements
    """
    if type(oneormoreitems) == list:
        return oneormoreitems
    elif type(oneormoreitems) == tuple:
        return [i for i in oneormoreitems]
    else:
        return [oneormoreitems]


def set_get_attr(obj, attr, value=None):
    """
    checks if attribute exists, if not, creates attribute with value None

    Parameters
    ----------
        obj: object
        attr: str

    Returns
    -------
        value(obj.attr)
    """
    if not hasattr(obj, attr):
        setattr(obj, attr, value)
    return getattr(obj, attr)


def append_if_not_exists(elist, element, operation):
    """
    appends an element to a list if it does not exist in list
    :param elist:
    :param element:
    :param operation:
    :return:
    """
    if operation == 'append':
        if not element in elist:
            elist.append(element)
    if operation == 'remove':
        if element in elist:
            elist.remove(element)
    return elist


def get_common_dtypes_from_list(mlist):
    """
    Takes a list of measurements and returns a list of common dtypes.

    Example
    -------
       mlist = [hys(down_field, up_field), hys(down_field, up_field, virgin)]
       returns ['down_field', 'up_field']

    Parameter
    ---------
       mlist: list
          list of measurements

    Returns
    -------
       dtypes
          sorted list of comment dtypes
    """
    # get intersection of all measurements with certain dtype
    dtypes = None
    if not mlist:
        raise ValueError('no measurements passed')
    for m in mlist:
        if not dtypes:
            dtypes = set(m.data.keys())
        else:
            dtypes = dtypes & set(m.data.keys())
    return sorted(list(dtypes))


@contextmanager
def ignored(*exceptions):
    try:
        yield
    except exceptions:
        pass


def get_full_argspec(func, args=None, kwargs=None):
    argspec = {}
    signature = inspect.signature(func)

    for i, v in enumerate(signature.parameters):
        try:
            # sets the value to the passed argument
            argspec[v] = args[i]
        except IndexError:  # no arg has been passed
            if not isinstance(signature.parameters[v].default, inspect._empty):
                argspec[v] = signature.parameters[v].default
    argspec.update(kwargs)
    return argspec


class plot(object):
    def __init__(self, single=False, result_feature=False, mtypes='none',
                 update_lims=True, overwrite_mobj_plt_props=None):  # todo define axis here? second_y?
        """
        If there are decorator arguments, the function
        to be decorated is not passed to the constructor!

        Parameter
        ---------
            mtypes: list(tuples)
                A list or list of tuples
            update_lims: bool
                default: True
                    if True updates both xlim and xlims to best fit the data
                    if False does not change the limits, might cause data to not be displayed
            result_feature:
                default: False
                    for features that require a certain result to be calculated beforehand

        Explanation:
        -----------
            the given mtype is used to filter all measurements to only use the ones where the feature is possible to
            produce.

            Example
            -------
                1. mtypes='hysteresis' will only use hysteresis measurements
                2. mtypes = ['armacq','trmacw' , 'irmacq'] can be calculated for ARM, TRM & IRM acquisition measurements
                    any of the three is available will be plotted
                3. mtypes = [('hysteresis', 'backfield')] needs a hysteresis and a backfield curve.
                    NOTE: The measurements need to have the same series!
                    #todo decide if we want to plot all possible or just the first combination.
                4. mtypes = [('mtype1', 'mtype2'),('mtype3','mtype4')] #todo cant think of an example but maybe possible

        """
        self.single = single
        self.result_feature = result_feature
        self.update_lims = update_lims
        self.mtypes = tuple2list_of_tuples(mtypes)
        if not overwrite_mobj_plt_props:
            overwrite_mobj_plt_props = {}
        self.overwrite_mobj_plt_props = overwrite_mobj_plt_props

    @staticmethod
    def short_feature_name(feature):
        return feature.__name__.replace('feature_', '')

    # @staticmethod
    def plt_single_feature(self, feature, visual, *args, **kwargs):
        """
        plotting of a single feature
        """
        visual.log.debug('PLOTTING SINGLE FEATURE: {}'.format(feature.__name__))

        # get all lines in visual.ax object BEFORE the feature is plotted
        old_lines = set(visual.ax.lines)
        feature(visual, *args, **kwargs)

        # get all NEW lines in visual.ax object AFTER the feature is plotted
        new_lines = [i for i in visual.ax.lines if i not in old_lines]
        visual.linedict.setdefault(feature.__name__, []).extend(new_lines)

    @staticmethod
    def get_plt_infos(feature, visual, name):
        """
        overwrites the plt_input, groupmean, samplemean, base, other, ignoresampe ect. bottom up.

        figure -> visual -> feature

        :param feature: the feature to be plotted (wrapped)
        :param visual: the visual the feature belongs to
        :param name: the name of the feature in case there are multiple
        :return:
        """
        types = ('figure', 'visual', 'feature')
        plt_input = {'in_type': None}

        for idx, indict in enumerate([visual.data, visual.features[name]['data']]):
                # [visual._RockPy_figure.fig_input, visual.data, visual.features[name]['data']]):
            if any(indict[i] for i in ('groupmean', 'samplemean', 'samplebase', 'groupbase', 'other')):
                plt_input.update(indict)
                plt_input.update({'in_type': types[idx]})

        # overwite any additional infos
        for info in ('base_alpha', 'ignore_samples', 'calculation_parameter', 'result_from_means',
                     'plot_groupmean', 'plot_samplemean', 'plot_groupbase', 'plot_samplebase', 'plot_other'):
            for idx, info_data in enumerate(
                    [getattr(visual._RockPy_figure, info), getattr(visual, info), visual.features[name][info]]):

                # calculation parameter are a dict -> they have to be updated
                if info == 'calculation_parameter':
                    plt_input.setdefault(info, {})
                    plt_input[info].update(info_data)
                # everything else has to be overwritten
                else:
                    plt_input.setdefault(info, None)
                    if info_data is not None:
                        plt_input[info] = info_data

        # set the default if nothing has been chosen
        for info in ('plot_groupmean', 'plot_samplemean',
                     'plot_groupbase', 'plot_samplebase',
                     'plot_other',
                     'result_from_means',):
            if plt_input[info] is None:
                plt_input[info] = True

        if not plt_input['base_alpha']:
            plt_input['base_alpha'] = 0.5

        return plt_input

    def update_plt_props(self, kwargs, visual, name):
        kwargs['plt_props'].update(self.overwrite_mobj_plt_props)
        kwargs['plt_props'].update(visual._RockPy_figure.plt_props)
        kwargs['plt_props'].update(visual.plt_props)
        kwargs['plt_props'].update(visual.features[name]['feature_props'])
        return kwargs

    def __call__(self, feature, name=None):
        """
        If there are decorator arguments, __call__() is only called
        once, as part of the decoration process! You can only give
        it a single argument, which is the function object.
        """

        def wrapped_feature(*args, **kwargs):
            # format the argspec
            parameter = get_full_argspec(func=feature, args=args, kwargs=kwargs)
            visual = parameter['self']

            # update the plt_props of the feature
            kwargs['plt_props'] = {}

            name = kwargs.pop('name', '')

            if self.single:
                self.update_plt_props(kwargs, visual=visual, name=name)
                self.plt_single_feature(feature=feature, visual=visual, **kwargs)
                return wrapped_feature

            ############################################################################################################
            # determine data parameters hierachically from fig -> visual -> feature
            # meaning if only fig data, plot fig_input
            # if fig_input and data -> fig_input is overwritten by data etc.
            plt_info = self.get_plt_infos(feature=feature, visual=visual, name=name)
            RockPy3.logger.debug('Input from:')

            for k, v in sorted(plt_info.items()):
                RockPy3.logger.debug('            {}:{}'.format(k, v))

            ############################################################################################################
            # RESULT FEATURES

            if self.result_feature:
                """
                result feature uses either visual of feature plot_mean, plot_base and ignore_samples flags
                if ignore_samples True: the mean of all samples_is calculated
                                  False: the mean for each individual sample is plotted
                """
                # cycle through possible inputs
                data = {}
                for plt_type in ('groupbase', 'samplebase', 'other', 'samplemean', 'groupmean'):
                    # skip everything that should not be plotted
                    if not plt_info['plot_' + plt_type]:
                        continue
                    # get the list of measurements
                    mlist = plt_info[plt_type]
                    # initialize plot properties
                    kwargs['plt_props'] = {}

                    # skip anything that has nothing to plot
                    if not mlist:
                        continue

                    # get rid of measurements without proper result and series
                    mlist = [m for m in mlist if m.has_result(visual.result) if m.get_series(stype=visual.series)]
                    # seaparate list into separate lists for each sample
                    # if ignore samples, all measurements will be used
                    if not plt_info['ignore_samples']:
                        samples = set(m.sobj.name for m in mlist)
                        mlists = [[m for m in mlist if m.sobj.name == sname] for sname in samples]
                    else:
                        mlists = [mlist]

                    # mlists is now a list of measurements for each sample
                    # iterate over the samples measurements
                    for mlist in mlists:
                        data.setdefault(plt_type, [])

                        #initialize the RPdata for each sample
                        sdata = RockPy3.Data(column_names=[visual.series, visual.result], row_names=[])

                        # calculate the results and get the series for all measurements
                        for m in mlist:
                            # get the sval of the specified series
                            sval = m.get_series(visual.series)[0].v
                            # calculate the result using the calculation parameters
                            res = getattr(m, 'result_' + visual.result)(**plt_info['calculation_parameter'])
                            d = RockPy3.Data(data=[sval, res[0]],
                                             column_names=[visual.series, visual.result],
                                             row_names='{}({})'.format(m.sobj.name, m.id))
                            sdata = sdata.append_rows(data=d)

                            # plot each individual result, append them to data
                            if 'base' in plt_type or plt_type == 'other':
                                kwargs['plt_props'] = deepcopy(m.plt_props)
                                # overwrite the plot properties of the measurement object if specified in the decorator
                                # e.g. for setting marker = '' in the hysteresis_data feature
                                self.update_plt_props(kwargs, visual=visual, name=name)

                                # update the plt_props for reducing the alpha
                                if (plt_type == 'samplebase' and plt_info['plot_samplemean'] and plt_info['samplemean']) or \
                                        (plt_type == 'groupbase' and plt_info['plot_groupmean' and plt_info['groupmean']]):
                                    kwargs['plt_props']['alpha'] = plt_info['base_alpha']
                                feature(visual, data=d, **kwargs)
                        else:
                            data[plt_type].append(sdata)

                        if plt_type == 'other' and plt_info['plot_samplemean']:
                            for i, sdata in enumerate(data['other']):
                                kwargs['plt_props'] = deepcopy(mlists[i][0].plt_props)
                                self.update_plt_props(kwargs, visual=visual, name=name)
                                sdata = sdata.eliminate_duplicate_variable_rows(substfunc='mean').sort()
                                # print(name)
                                # print(sdata)
                                feature(visual, data=sdata, **kwargs)

                    if 'mean' in plt_type:
                        mean_data = []
                        # if result_from_means: the results will be calculated from the data  of any mean measurement
                        # that has been calculated beforand otherwise the mean of the results will be calculated
                        if not plt_info['result_from_means']:
                            base = plt_type.replace('mean', 'base')
                            for sdata in data[base]:
                                sdata = sdata.eliminate_duplicate_variable_rows(substfunc='mean').sort()
                                mean_data.append(sdata)
                        else:
                            for sdata in data[plt_type]:
                                sdata = sdata.eliminate_duplicate_variable_rows(substfunc='mean').sort()
                                mean_data.append(sdata)
                        # print(mean_data)
                        for sdata in mean_data:
                            # print(sdata)
                            kwargs['plt_props'] = {}
                            kwargs['plt_props'] = deepcopy(mlist[0].plt_props)

                            # overwrite the plot properties of the measurement object if specified in the decorator
                            # e.g. for setting marker = '' in the hysteresis_data feature
                            self.update_plt_props(kwargs, visual=visual, name=name)
                            kwargs['plt_props'].update({'alpha': 1, 'zorder': 100})
                            feature(visual, data=sdata, **kwargs)

            ############################################################################################################
            # NORMAL FEATURES
            else:
                # cycle through possible inputs
                for plt_type in ('groupmean', 'samplemean', 'groupbase', 'samplebase', 'other'):
                    # skip everything that should not be plotted
                    if not plt_info['plot_' + plt_type] or not plt_type in plt_info:
                        continue
                    # get the list of measurements
                    mlist = plt_info[plt_type]
                    # initialize plot properties
                    kwargs['plt_props'] = {}

                    # for visuals with several possible mtypes
                    for mtype in self.mtypes:
                        if type(mtype) == str:
                            mtype = (mtype,)
                        if len(mtype) > 1:
                            mobj = MlistToTupleList(mlist, mtype)
                            # print(mobj)
                        elif len(mtype) == 1:
                            mobj = [m for m in mlist if m.mtype == mtype[0]]
                        else:
                            visual.log.error(
                                'FEATURE {} data doesnt match mtype requirements {}'.format(feature.__name__,
                                                                                             mtype))
                        for mt_tuple in mobj:
                            try:
                                kwargs['plt_props'].update(mt_tuple[0].plt_props)
                            except TypeError:
                                kwargs['plt_props'].update(mt_tuple.plt_props)

                            # overwrite the plot properties of the measurement object if specified in the decorator
                            # e.g. for setting marker = '' in the hysteresis_data feature
                            self.update_plt_props(kwargs, visual=visual, name=name)

                            # change the alpha to base_alpha if group or sample base should be plotted and the means
                            if (plt_type == 'samplebase' and plt_info['plot_samplemean'] and plt_info['samplemean']) or \
                                    (plt_type == 'groupbase' and plt_info['plot_groupmean'] and plt_info['groupmean']):
                                kwargs['plt_props']['alpha'] = plt_info['base_alpha']
                            if plt_type == 'groupmean':
                                kwargs['plt_props']['zorder'] = 100
                            if plt_type == 'samplemean':
                                kwargs['plt_props']['zorder'] = 50
                            feature(visual, mobj=mt_tuple, **kwargs)

        return wrapped_feature


def MlistToTupleList(mlist, mtypes, ignore_series=False):
    """
    Transforma a list of measurements into a tuplelist, according to the mtypes specified.

    Parameter
    ---------
        mlist: list
            list of RockPy Measurements
        mtypes: tuple
            tuple for the mlist to be organized by.

    Example
    -------
        1. multiple mtypes (tuple)
            mlist = [Hys(S1), Hys(S2), Coe(S1), Coe(S2)] assuming that S1 means all series are the same
            mtypes = (hys, coe)

            1. the list is sorted into a dictionary with mtype:list(m)
            2. for each member of dict[hys] a corresponding coe measurement is searched which has
                a) the same parent sample
                b) exactly the same series


    """
    # create the dictionary
    mdict = {mtype: [m for m in mlist if m.mtype == mtype] for mtype in mtypes}

    out = []
    for m_mtype1 in mdict[mtypes[0]]:
        aux = [m_mtype1]
        for mtype in mtypes[1:]:
            for m_mtype_n in mdict[mtype]:
                if not m_mtype1.sample_obj == m_mtype_n.sample_obj:
                    break
                if compare_measurement_series(m_mtype1, m_mtype_n) or ignore_series:
                    aux.append(m_mtype_n)
                    if not ignore_series:
                        break
        out.append(tuple(aux))
    return out


def compare_measurement_series(m1, m2):
    """
    returns True if both series have exactly the same series.

    Parameter
    ---------
        m1: RockPy.Measeurement
        m2: RockPy.Measeurement

    Note
    ----
        ignores multiples of the same series
    """
    s1 = m1.series
    s2 = m2.series

    if all(s in s2 for s in s1) and all(s in s1 for s in s2):
        return True
    else:
        return False

def separate_cparam_str(**kwargs):

    # the kwargs need to be sorted so that they follow the hierarchy
    # 1. parameter only
    # 2. mtype & parameter
    # 3. method & parameter
    # 4. mtype & method & parameter

    param_only  = [i for i in kwargs if [i] == i.split('___') if [i] == i.split('__')]
    mtype_only  = [i for i in kwargs if [i] == i.split('___') if [i] != i.split('__')]
    method_only = [i for i in kwargs if [i] != i.split('___') if [i] == i.split('__')]
    mixed = [i for i in kwargs if [i] != i.split('__') if [i] != i.split('___')]

    print(param_only, mtype_only, method_only, mixed)

if __name__ == '__main__':
    separate_cparam_str(saturation_percent=10, hys__saturation_percent=10, bc___saturation_percent=10, hys___bc__saturation_percent=10)

def kwargs_to_calculation_parameter(rpobj=None, mtype_list=None, result=None, **kwargs):
    """
    looks though all provided kwargs and searches for possible calculation parameters. kwarg key naming see:
    separate_mtype_method_parameter

    Parameters
    ----------
        rpobj: RockPy Object (Study SampleGroup, Sample, Measurement
            default: None
        mtypes: list
            list of possible mtypes, parameters are filtered for these
        result: str
            parameters are filtered

    Hierarchy
    ---------
        4. passing a pure parameter causes all methods in all mtypes to be set to that value
           !!! this will be overwritten by !!!
        3. passing a mtype but no method
           !!! this will be overwritten by !!!
        2. passing a method
           !!! this will be overwritten by !!!
        1. passing a method and mtype
           !!! this will be overwritten by !!!

    Note
    ----
        passing a mtype:
            will add mtype: { all_methods_with_parameter: {parameter:parameter_value}} to calculation_parameter dict
        passing a method:
            will add all_mtypes_with_method: { method: {parameter:parameter_value}} to calculation_parameter dict
        passing mtype and method:
            will add all_mtypes_with_method: { all_methods_with_parameter: {parameter:parameter_value}} to calculation_parameter dict

    Returns
    -------
        calculation_parameter: dict
            dictionary with the parameters passed to the method, where a calculation_method could be found
        kwargs: dict

    computation notes:
        if only parameter is specified: look for all mtypes with the parameter and all methods with the parameter
        if mtype specified: look for all methods with the parameter
    """

    # the kwargs need to be sorted so that they follow the hierarchy
    # 1. parameter only
    # 2. mtype & parameter
    # 3. method & parameter
    # 4. mtype & method & parameter

    param_only = [i for i in kwargs if [i] == i.split('___') if [i] == i.split('__')]
    mtype_only = [i for i in kwargs if [i] == i.split('___') if [i] != i.split('__')]
    method_only = [i for i in kwargs if [i] == i.split('__') if [i] != i.split('___')]
    mixed = [i for i in kwargs if [i] != i.split('__') if [i] != i.split('___')]

    kwarg_list = param_only + mtype_only + method_only + mixed

    calc_params = {}

    for kwarg in kwarg_list:
        remove = False
        mtypes, methods, parameter = RockPy3.core.utils.separate_mtype_method_parameter(kwarg=kwarg)

        # get all mtypes, methods if not specified in kwarg
        # nothing specified
        if not mtypes and not methods:
            mtypes = [mtype for mtype, params in RockPy3.Measurement.collected_mtype_calculation_parameter().items() if
                      parameter in params]

            # filter only given in mtype_list
            if mtype_list:
                mtypes = [mtype for mtype in mtypes if mtype in mtype_list]

        # no mtype specified
        elif not mtypes:
            # we need to add methods with recipes:
            # bc___recipe = 'simple' would otherwise not be added because there is no method calculate_bc
            for method in methods:
                for calc_method, method_params in RockPy3.Measurement.method_calculation_parameter_list().items():
                    if calc_method.split('_')[-1].isupper() and ''.join(calc_method.split('_')[:-1]) == method:
                        methods.append(calc_method)

            mtypes = [mtype for mtype, mtype_methods in RockPy3.Measurement.mtype_calculation_parameter().items()
                      if any(method in mtype_methods.keys() for method in methods)]
            # filter only given in mtype_list
            if mtype_list:
                mtypes = [mtype for mtype in mtypes if mtype in mtype_list]

        if not methods:
            methods = [method for method, params in RockPy3.Measurement.method_calculation_parameter_list().items()
                       if
                       parameter in params]

        # if an object is given, we can filter the possible mtypes, and methods further
        # 1. a measurement object
        if isinstance(rpobj, RockPy3.Measurement):
            mtypes = [mtype for mtype in mtypes if mtype == rpobj.mtype]
            methods = [method for method in methods if method in rpobj.mtype_possible_calculation_parameter()]
            # print(mtypes, methods, parameter)

        # 2. a sample object
        if isinstance(rpobj, RockPy3.Sample):
            mtypes = [mtype for mtype in mtypes if mtype == rpobj.mtypes]

        # 3. a samplegroup object
        # 4. a study object
        # 5. a visual object
        # 6. a result
        if result:
            methods = [result]

        ############################################################################################################
        # actual calculation
        for mtype in mtypes:
            # get the only  methods that are implemented in the mtype to be checked
            check_methods = set(RockPy3.Measurement.mtype_calculation_parameter()[mtype]) & set(methods)

            for method in check_methods:
                # with ignored(KeyError): # ignore keyerrors if mtype / method couple does not match
                try:
                    if parameter in RockPy3.Measurement.mtype_calculation_parameter()[mtype][method]:
                        RockPy3.logger.debug('PARAMETER found in << %s, %s >>' % (mtype, method))
                        remove = True
                        calc_params.setdefault(mtype, dict())
                        calc_params[mtype].setdefault(method, dict())
                        calc_params[mtype][method].update({parameter: kwargs[kwarg]})
                    else:
                        RockPy3.logger.error(
                            'PARAMETER << %s >> NOT found in << %s, %s >>' % (parameter, mtype, method))
                except KeyError:
                    RockPy3.logger.debug(
                        'PARAMETER << %s >> not found mtype, method pair probably wrong << %s, %s >>' % (
                            parameter, mtype, method))
        if remove:
            kwargs.pop(kwarg)

    return calc_params, kwargs


def sort_input(input):
    """
    sorts a inputlist into subcategories
    :param input:
    :param groupmean:
    :param samplemean:
    :param base:
    :param other:
    :return:
    """
    out = dict(groupmean=set(), samplemean=set(), groupbase=set(), samplebase=set(), other=set())

    if not input:
        return out

    # get all measurements in plt_input
    mlist, meanlist = mlist_from_input(input)

    if not any([mlist, meanlist]):
        RockPy3.logger.warning(
            'No data selected groupmean, samplemean, base and other will only work for the speecific data of figure visual or feature.')

    # group_means
    group_means = set(m for m in meanlist if isinstance(m.sobj, RockPy3.MeanSample))
    group_base_ids = set(id for m in group_means for id in m.base_ids)
    out['groupmean'] = group_means
    out['groupbase'] = set(m for m in mlist + meanlist if m.id in group_base_ids)

    # sample_means
    sample_means = set(m for m in meanlist if type(m.sobj) == RockPy3.Sample)
    out['samplemean'] = sample_means
    sample_base_ids = set(id for m in sample_means for id in m.base_ids)
    out['samplebase'] = set(m for m in mlist if m.id in sample_base_ids)

    # get all ids for an base measurement including group means
    all_base_ids = set(id for m in meanlist for id in m.base_ids)

    # other are all measurements that are not in the base id set
    other = set(m for m in mlist if not m.id in all_base_ids)
    out['other'] = other

    for k in out:
        out[k] = sorted(out[k])
    return deepcopy(out)


def add_to_input(input, to_add_to, groupmean=True, samplemean=True, base=True, other=True):
    input = sort_input(input=input)
    for k, v in to_add_to.items():
        to_add_to[k].update(input[k])
    return to_add_to


def mlist_from_input(plt_input):
    """
    takes arbitrary data and separates all measurements.

    explanation:
        sample - measurement sorted to mlist
        sample - meanmeasurement sorted to meanlist
        meansample - measurement sorted to meanlist
    :param plt_input:
    :return:
    """
    mlist = []
    meanlist = []

    plt_input = to_list(plt_input)

    for item in plt_input:
        if isinstance(item, RockPy3.RockPyStudy):
            plt_input.extend(item.samplelist)
        if isinstance(item, RockPy3.Sample):
            mlist.extend(item.measurements)
            meanlist.extend(item.mean_measurements)
        if isinstance(item, RockPy3.MeanSample):
            meanlist.extend(item.measurements)
        if isinstance(item, RockPy3.Measurement):
            if item.is_mean:
                meanlist.append(item)
            else:
                mlist.append(item)
    return mlist, meanlist


def compare_measurement_series(m1, m2):
    """
    returns True if both series have exactly the same series.

    Parameter
    ---------
        m1: RockPy3.Measeurement
        m2: RockPy3.Measeurement

    Note
    ----
        ignores multiples of the same series
    """
    s1 = m1.series
    s2 = m2.series

    if all(s in s2 for s in s1) and all(s in s1 for s in s2):
        return True
    else:
        return False


def MlistToTupleList(mlist, mtypes):
    """
    Transforma a list of measurements into a tuplelist, according to the mtypes specified.

    Parameter
    ---------
        mlist: list
            list of RockPy Measurements
        mtypes: tuple
            tuple for the mlist to be organized by.

    Example
    -------
        mlist = [Hys(S1), Hys(S2), Coe(S1), Coe(S2)] assuming that S1 means all series are the same
        mtypes = (hys, coe)

        1. the list is sorted into a dictionary with mtype:list(m)
        2. for each member of dict[hys] a corresponding coe measurement is searched which has
            a) the same parent sample
            b) exactly the same series
    """
    # create the dictionary
    mdict = {mtype: [m for m in mlist if m.mtype == mtype] for mtype in mtypes}
    out = []

    for m_mtype1 in mdict[mtypes[0]]:
        # print m_mtype1
        aux = [m_mtype1]
        for mtype in mtypes[1:]:
            for m_mtype_n in mdict[mtype]:
                if not m_mtype1.sobj == m_mtype_n.sobj:
                    continue
                if compare_measurement_series(m_mtype1, m_mtype_n):
                    aux.append(m_mtype_n)
                    continue
        out.append(tuple(aux))
    return out

def compare_measurement_series(m1, m2):
    """
    returns True if both series have exactly the same series.
    Parameter
    ---------
        m1: RockPy.Measeurement
        m2: RockPy.Measeurement
    Note
    ----
        ignores multiples of the same series
    """
    s1 = m1.series
    s2 = m2.series

    if all(s in s2 for s in s1) and all(s in s1 for s in s2):
        return True
    else:
        return False


def separate_mtype_method_parameter(kwarg):
    """
    separetes the possible kwarg parameters for calculation_parameter lookup.
    the mtype has to be followed by 3x_ the methodname by 2x _ because calculation_parameter may be separated by 1x_

    Format
    ------
        mtype__method___parameter or multiples of mtyep, method
        mtype1__mtype2__method1___method2___parameter but always only one parameter LAST
        mtype1__method1___mtype2__method2___parameter mixed also possible


    Retuns
    ------
        mtype___method__parameter: [mtype], [method], parameter
        mtype___parameter: [mtype], [], parameter
        method__parameter: [], [method], parameter
        parameter: [], [], parameter
    """
    method = []
    mtype = []
    parameter = None

    # all possible methods, mtypes and the parameter
    possible = [j for i in kwarg.split('___') for j in i.split('__')]

    for i in possible:
        # remove the part
        kwarg = kwarg.replace(i, '')
        if kwarg.startswith('___'):
            method.append(i)
            kwarg = kwarg[3:]
        if kwarg.startswith('__'):
            mtype.append(i)
            kwarg = kwarg[2:]

    parameter = possible[-1]
    return mtype, method, parameter


def separate_calculation_parameter_from_kwargs(rpobj=None, mtype_list=None, **kwargs):
    """
    separates kwargs from calcuzlation arameters, without changing the signature
        e.g. hysteresis__no_points = n !-> hystersis:{no_points: n}

    """
    calculation_parameter, non_calculation_parameter = kwargs_to_calculation_parameter(rpobj=rpobj,
                                                                                       mtype_list=mtype_list,
                                                                                       **kwargs)
    out = {}

    for key, value in kwargs.items():
        if not key in non_calculation_parameter:
            out.setdefault(key, value)

    return out, non_calculation_parameter


def tuple2list_of_tuples(item):
    """
    Takes a list of tuples or a tuple and returns a list of tuples

    Parameters
    ----------
       data: list, tuple

    Returns
    -------
       list
          Returns a list of tuples, if data is a tuple it converts it to a list of tuples
          if data == a list of tuples will just return data
    """
    if type(item) != tuple and type(item) != list:
        item = tuple([item])
    if type(item) == tuple:
        aux = list()
        aux.append(item)
        item = aux
    return item

def range_to_tuple(range):
    seperators = ('<', '=', '>')

    out = []
    if any(s in range for s in seperators):
        if range.startswith('<'):
            pass
    else:
        return float(range)
#
# if __name__ == '__main__':
#     Study = RockPy3.RockPyStudy()
#     # sample
#     s2 = Study.add_sample(name='S2')
#     # noise ranging from 0-5%
#     for noise in range(5):
#         # 4 measurements for each noise
#         for n in range(4):
#             h1 = s2.add_simulation(mtype='hysteresis', noise = noise, marker='o')
#             # we add a series for the noise
#             h1.add_series('noise', noise, '%')
#
#     fig = RockPy3.Figure(fig_input=Study)
#     v = fig.add_visual(visual='hysteresis')
#     v = fig.add_visual(visual='resultseries',result='ms', series='noise')
#     v = fig.add_visual(visual='resultseries',result='bc', series='noise')
#     v.add_feature('result_series_errorbars')
#     fig.show()


def _to_tuple(oneormoreitems):
    """
    convert argument to tuple of elements
       oneormoreitems: single number or string or list of numbers or strings
    :return: tuple of elements
    """
    return tuple(oneormoreitems) if hasattr(oneormoreitems, '__iter__') and type(oneormoreitems) is not str else (oneormoreitems, )

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