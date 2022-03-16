from __future__ import print_function
from __future__ import division
from . import _C

import fuzzytools.strings as strings
import fuzzytools.matplotlib.colors as ftc
import numpy as np
import fuzzytools.files as ftfiles

INVALID_MODEL_KEYS = [
	'method',
	'tmode',
	]
MODEL_KEYS_REPLACE = {
	}

###################################################################################################################################################

def get_model_names(rootdir, cfilename, kf, lcset_name):
	roodirs = [r.split('/')[-1] for r in ftfiles.get_roodirs(rootdir)]
	model_names = [r for r in roodirs if '~' in r]
	model_names.sort()
	return model_names

def get_fmodel_name(model_name,
	returns_mn_dict=False,
	):
	mn_dict = strings.get_dict_from_string(model_name)
	if not 'training-set' in mn_dict.keys():
		train_mode = mn_dict['tmode']
		method = mn_dict['method']
		mn_dict['training-set'] = f'[{train_mode}]'if train_mode=='r' else f'{method}[{train_mode}]'
	mdl_info = []
	for k in mn_dict.keys():
		if k in INVALID_MODEL_KEYS:
			continue
		mdl_info += [f'{MODEL_KEYS_REPLACE.get(k, k)}={mn_dict[k]}']
	mdl_info = '; '.join(mdl_info)
	fmodel_name = f'BRF ({mdl_info})'
	if returns_mn_dict:
		return fmodel_name, mn_dict
	else:
		return fmodel_name

def _get_unique_model_name(model_name):
	_, mn_dict = get_fmodel_name(model_name, returns_mn_dict=True)
	unique_model_name = get_fmodel_name(strings.get_string_from_dict(mn_dict))
	unique_model_name = unique_model_name.replace('[r+s]', '').replace('[s]', '').replace('[r]', '')
	return unique_model_name

def get_model_color(target_model_name, model_names):
	unique_model_names = []
	for model_name in model_names:
		unique_model_name = _get_unique_model_name(model_name)
		if not unique_model_name in unique_model_names:
			unique_model_names += [unique_model_name]

	color_dict = ftc.get_color_dict(unique_model_names)
	color = color_dict[_get_unique_model_name(target_model_name)]
	return color
