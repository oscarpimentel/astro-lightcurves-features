from __future__ import print_function
from __future__ import division
from . import _C

import pandas as pd
from fuzzytools.files import create_dir
import os
import numpy as np
from copy import copy, deepcopy

BAND_NAMES = ['g', 'r']
SNE_SELECTED_FEATURES = _C.SNE_SELECTED_FEATURES
FATS_MODE = 'all'

###################################################################################################################################################

def propagate_spm_nans(_df_x,
	band_names=BAND_NAMES,
	):
	df_x = copy(_df_x)
	for idx, row in df_x.iterrows():
		for b in band_names:
			spm_b_columns = [c for c in df_x.columns if 'SPM' in c and c[-1]==b]
			spm_b = row[spm_b_columns]
			has_nan = np.any(np.isnan(spm_b.values))
			if has_nan:
				for c in spm_b_columns:
					df_x.loc[idx,c] = np.nan
	return df_x

def get_multiband_features(invalid_features,
	band_names=BAND_NAMES,
	):
	x = []
	for i in invalid_features:
		for b in band_names:
			x.append(f'{i}_{b}')
	return x

def load_features(filedir,
	fats_mode=FATS_MODE,
	thday=100,
	):
	thdays_features_df = pd.read_parquet(os.path.abspath(f'{filedir}')) # parquet
	thday_features_df = thdays_features_df.loc[thdays_features_df['_thday']==thday]
	columns = list(thday_features_df.columns)
	y_columns = [c for c in columns if '_'==c[0]]
	df_x = thday_features_df[[c for c in columns if not c in y_columns]]
	df_y = thday_features_df[y_columns]

	if fats_mode=='all':
		pass

	if fats_mode=='sne':
		query_features = get_multiband_features(SNE_SELECTED_FEATURES)
		invalid_features = get_multiband_features([])
		df_x = df_x[[c for c in df_x.columns if c in query_features and not c in invalid_features]]

	if fats_mode=='spm':
		query_features = get_multiband_features([f for f in SNE_SELECTED_FEATURES if 'SPM' in f])
		invalid_features = get_multiband_features([])
		df_x = df_x[[c for c in df_x.columns if c in query_features and not c in invalid_features]]
	
	df_x = df_x.astype(np.float32)
	return df_x, df_y

def save_features(thdays_features_df, save_filedir):
	save_rootdir = '/'.join([s for s in save_filedir.split('/')[:-1]])
	create_dir(save_rootdir)
	thdays_features_df.to_parquet(os.path.abspath(f'{save_filedir}')) # parquet