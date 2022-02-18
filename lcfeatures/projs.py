from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
from .files import load_features
from fuzzytools.dataframes import clean_df_nans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA, KernelPCA, FastICA
from umap import UMAP
from umap.parametric_umap import ParametricUMAP
from fuzzytools.datascience.dim_reductors import DimReductorPipeline
import tensorflow as tf

NAN_MODE = 'mean' # value mean median
RANDOM_STATE = None

###################################################################################################################################################

def get_fitted_2dproj(lcdataset, s_lcset_name, load_rootdir, proj_mode, features_mode,
	random_state=RANDOM_STATE,
	supervised=False, # True False
	):
	r_lcset_name = s_lcset_name.split('.')[0]
	r_lcset = lcdataset[r_lcset_name]

	r_df_x, r_df_y = load_features(f'{load_rootdir}/{r_lcset_name}.df', features_mode)
	s_df_x, s_df_y = load_features(f'{load_rootdir}/{s_lcset_name}.df', features_mode)

	print(r_df_x.values.shape)
	df_x = pd.concat([r_df_x, s_df_x], axis='rows')
	df_x, _, _ = clean_df_nans(df_x, mode=NAN_MODE, drop_null_columns=True)
	x = df_x.values
	y = np.concatenate([r_df_y[['_y']].values[...,0], s_df_y[['_y']].values[...,0]], axis=0)
	r_lcobj_names = r_df_y[['_lcobj_name']].values[...,0].tolist()
	s_lcobj_names = s_df_y[['_lcobj_name']].values[...,0].tolist()
	map_lcobj_names = r_lcobj_names+s_lcobj_names

	if proj_mode=='PCA':
		dim_reductor = DimReductorPipeline([
			QuantileTransformer(output_distribution='normal'),
			StandardScaler(),
			PCA(n_components=2),
			])
		dim_reductor.fit(x,
			reduction_map_kwargs={'y':y} if supervised else None,
			)
		map_x = dim_reductor.transform(x)

	elif proj_mode=='UMAP':
		dim_reductor = DimReductorPipeline([
			QuantileTransformer(output_distribution='normal'),
			StandardScaler(),
			PCA(n_components=10),
			UMAP(
				n_components=2,
				metric='euclidean',
				n_neighbors=10,
				min_dist=.01,
				random_state=random_state,
				transform_seed=random_state,
				),
			])
		dim_reductor.fit(x,
			reduction_map_kwargs={'y':y} if supervised else None,
			)
		map_x = dim_reductor.transform(x)

	else:
		raise Exception(f'proj_mode={proj_mode}')

	d = {
		'proj_mode':proj_mode,
		'map_lcobj_names':map_lcobj_names,
		'map_x':map_x,
		'y':y,
		'class_names':r_lcset.class_names,
		'r_lcset_name':r_lcset_name,
		's_lcset_name':s_lcset_name,
		'r_lcobj_names':r_lcobj_names,
		's_lcobj_names':s_lcobj_names,
	}
	return d