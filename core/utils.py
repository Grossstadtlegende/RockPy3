__author__ = 'mike'
from contextlib import contextmanager
import RockPy3
import logging

def create_logger(name):
    log = logging.getLogger(name=name)
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-10s %(name)-20s %(message)s', "%H:%M:%S")
    # formatter = logging.Formatter('%(asctime)s: %(levelname)-10s %(name)-20s %(message)s')
    # fh = logging.FileHandler('RPV3.log')
    # fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    # ch.setLevel(logging.WARNING)
    ch.setLevel(logging.NOTSET)
    ch.setFormatter(formatter)
    # log.addHandler(fh)
    log.addHandler(ch)

    return log  # ch#, fh

def to_list(oneormoreitems):
    """
    convert argument to tuple of elements
    :param oneormoreitems: single number or string or list of numbers or strings
    :return: tuple of elements
    """
    return oneormoreitems if hasattr(oneormoreitems, '__iter__') else [oneormoreitems]

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


@contextmanager
def ignored(*exceptions):
    try:
        yield
    except exceptions:
        pass


class feature(object):
    def __init__(self, plt_frequency='multiple', mtypes='none', update_lims=True): #todo define axis here? second_y?
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
        self.plt_frequency = plt_frequency
        self.update_lims= update_lims
        self.mtypes = RockPy3.utils.general.tuple2list_of_tuples(mtypes)

    def plt_single_feature(self, feature, visual, *args, **kwargs):
        """
        plotting of a single feature
        """
        visual.logger.debug('PLOTTING SINGLE FEATURE: {}'.format(feature.__name__))

        # get all lines in visual.ax object BEFORE the feature is plotted
        old_lines = set(visual.ax.lines)
        #plot the feature
        feature(*args, **kwargs)

        # get all NEW lines in visual.ax object AFTER the feature is plotted
        new_lines = [i for i in visual.ax.lines if i not in old_lines]
        visual.linedict.setdefault(feature.__name__, []).extend(new_lines)

    def __call__(self, feature):
        """
        If there are decorator arguments, __call__() is only called
        once, as part of the decoration process! You can only give
        it a single argument, which is the function object.
        """
        def wrapped_feature(*args, **kwargs):
            # format the argspec
            parameter = RockPy3.utils.general.get_full_argspec(func=feature, args=args)

            visual = parameter['self']
            study, measurements_only = visual.get_virtual_study()

            if 'calculation_parameter' in parameter:
                calculation_parameter, kwargs = RockPy3.utils.general.separate_calculation_parameter_from_kwargs(mtype_list=self.mtypes, **kwargs)

            #get xlims for first axis
            xlims = deepcopy(visual.ax.get_xlim())
            ylims = deepcopy(visual.ax.get_ylim())

            # plot single features only once
            if self.plt_frequency == 'single':
                self.plt_single_feature(feature, visual, *args, **kwargs)

            # plot multiple features for each measurement
            else:
                for sg_idx, sample_group in enumerate(study):
                    if isinstance(sample_group, RockPy.SampleGroup) and sample_group.has_mean:
                        sample_group.set_plt_prop(prop='alpha', value=0.5)

                    for sample_idx, sample in enumerate(sample_group):
                        for mtype in self.mtypes:

                            # if only measurements have been passed to the visual
                            if measurements_only:
                                measurements = [m for m in sample if m.mtype in mtype]
                            # otherwise search though the sample fo the correct measurements
                            else:
                                if isinstance(sample, RockPy3.MeanSample):
                                    measurements = sample.get_measurements(mtypes=mtype, mean=True)
                                else:
                                    measurements = sample.get_measurements(mtypes=mtype)

                            measurements = RockPy3.utils.general.MlistToTupleList(measurements, mtypes=mtype)

                            if not measurements:
                                visual.logger.error('CANT find/generate correct measurements tuple')
                                break

                            for m_idx, m in enumerate(measurements):
                                visual.logger.debug(
                                    'PLOTTING FEATURE: {} with measurement {}'.format(feature.__name__, ', '.join(i.mtype for i in m)))

                                # update single measurement if only one mtype
                                if len(mtype) == 1:
                                    mobj = m[0]
                                else:
                                    mobj = m

                                # change the mobj colors, ls and markers if not previously specified
                                visual.set_mobj_plt_props(m[0], [sg_idx, sample_idx, m_idx])

                                #set the measurement that should be plottet
                                kwargs.update(dict(mobj=mobj))

                                old_lines = set(visual.ax.lines)
                                # calculate the feature
                                feature(*args, **kwargs)

                                # extract the new lines generated by feature
                                new_lines = [i for i in visual.ax.lines if i not in old_lines]

                                for mobj in m:
                                    # finally add the sample, mobj,
                                    visual.vdict['sample'].setdefault(mobj.sample_obj.name, []).append(feature.__name__)
                                    visual.vdict['mobj'].setdefault(mobj, []).append(feature.__name__)
                                    visual.vdict['visual'].setdefault(feature.__name__, []).append(mobj)

                                    visual.linedict.setdefault(mobj, []).extend(new_lines)
                                    visual.linedict.setdefault(feature.__name__, []).extend(new_lines)

            # reset xlims and ylims to old limits
            if not self.update_lims:
                visual.ax.set_xlim(xlims)
                visual.ax.set_ylim(ylims)

        return wrapped_feature