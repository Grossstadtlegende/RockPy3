__author__ = 'mike'
import single_moment
from RockPy.core.data import RockPyData


class Trm(single_moment.generic_moment):
    def __init__(self, sample_obj,
                 mtype, fpath, ftype,
                 **options):

        super(Trm, self).__init__(sample_obj,
                                  mtype, fpath, ftype,
                                  **options)

    def format_cryomag(self):
        super(Trm, self).format_cryomag()
        self._raw_data['data'].rename_column('step', 'temp')
