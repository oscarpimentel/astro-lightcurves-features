from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import fuzzytools.strings as strings
import fuzzytools.files as ftfiles
from fuzzytools.datascience.xerror import XError
from fuzzytools.dataframes import DFBuilder
from fuzzytools.latex.latex_tables import LatexTable
import matplotlib.pyplot as plt
from . import utils as utils

RANDOM_STATE = None
METRICS_D = _C.METRICS_D
DICT_NAME = 'thdays_class_metrics'

###################################################################################################################################################

def get_performance_df(rootdir, cfilename, kf, set_name, model_names, metric_names,
	target_class=None,
	thday=None,
	dict_name=DICT_NAME,
	):
	info_df = DFBuilder()
	new_model_names = model_names
	for kmn,model_name in enumerate(new_model_names):
		model_name_d = strings.get_dict_from_string(model_name)
		method = model_name_d['method']
		load_roodir = f'../save/{model_name}/performance/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method={method}'
		print(load_roodir)
		files, files_ids, kfs = ftfiles.gather_files_by_kfold(load_roodir, kf, set_name,
			fext='d',
			imbalanced_kf_mode='oversampling', # error oversampling
			random_state=RANDOM_STATE,
			)
		print(f'{files_ids}({len(files_ids)}#)')
		if len(files)==0:
			continue

		thdays = files[0]()['thdays']
		thday = thdays[-1] if thday is None else thday

		d = {}
		for metric_name in metric_names:
			new_metric_name = strings.latex_sub_superscript(METRICS_D[metric_name]['mn'],
				subscript=('\\text{'+target_class.replace('SN', '')+'}') if not target_class is None else ' ',
				)
			new_metric_name = 'b-'+new_metric_name if target_class is None else new_metric_name

			if target_class is None:
				xe_metric = XError([f()[f'{dict_name}_df'].loc[f()[f'{dict_name}_df']['_thday']==thday][f'b-{metric_name}'].item() for f in files])
			else:
				xe_metric = XError([f()[f'{dict_name}_cdf'][target_class].loc[f()[f'{dict_name}_df']['_thday']==thday][f'{metric_name}'].item() for f in files])
			d[new_metric_name] = xe_metric

		index = f'Model={utils.get_fmodel_name(model_name)}'
		info_df.append(index, d)

	return info_df