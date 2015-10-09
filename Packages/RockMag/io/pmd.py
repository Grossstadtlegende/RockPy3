__author__ = 'wack'

import numpy as np

from RockPy.core import io


class PMD(io.ftype):
    """
    pmd file format is used by PaleoMac (J.P. Cogne) as well as by PMGSC (Enkins)
    """

    def __init__(self, dfile, sample_name):
        super(PMD, self).__init__(dfile=dfile, sample_name=sample_name)

        # read the data file line by line
        f = open(self.file_name)
        self.raw_data = f.readlines()
        f.close()

        # store first header line
        self.header = self.raw_data[0]

        # get sample name and orientation data, organized in 10 character wide blocks + volume + comment from second line
        l = self.raw_data[1]
        # specify string indices of fields
        fidc = [0, 10, 20, 30, 40, 50, 63, None]
        fields = [l[fidc[i]:fidc[i+1]].strip() for i in xrange(len(fidc)-1)]
        sample_name = fields[0]
        if sample_name != self.sample_name:  # the sample name that we want to read and the one in the pmd file do not match
            self.log.warning('Sample name %s found in %s does not match %s' % (sample_name, self.file_name, self.sample_name))
            # break here ???

        # third line should be:
        # STEP  Xc (Am2)  Yc (Am2)  Zc (Am2)  MAG(A/m)   Dg    Ig    Ds    Is   a95
        # TODO: check???

        # only STEP, Xc, Yc, Zc and a95 are important, other columns can be recalculated
        # first character of STEP defines the measurement type i.e. N = NRM, A = AF, T = thermal

        # data of individual measurements starts from 4th line
        data = self.raw_data[3:]
        self.steptypes = [d[0:1] for d in data]  # first character of each line
        numdata = "\r\n".join([d[1:].strip() for d in data]) # rest of each line
        self.data = np.fromstring(numdata, sep=' ')



    def out_afdemag(self):
        # return field, x, y, z
        return self.data[:, 0:4]

