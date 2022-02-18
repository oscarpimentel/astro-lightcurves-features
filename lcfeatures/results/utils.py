from __future__ import print_function
from __future__ import division
from . import _C

import fuzzytools.strings as strings
import fuzzytools.matplotlib.colors as ftc
import numpy as np
import fuzzytools.files as ftfiles

###################################################################################################################################################

def get_model_names(rootdir, cfilename, kf, lcset_name):
	roodirs = [r.split('/')[-1] for r in ftfiles.get_roodirs(rootdir)]
	return [r for r in roodirs if '~' in r]