from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE
from fuzzytools.datascience.ranks import TopRank
import fuzzytools.datascience.metrics as fcm
from fuzzytools.dataframes import clean_df_nans
from fuzzytools.dataframes import DFBuilder
from fuzzytools.dicts import update_dicts
from .files import load_features

NAN_MODE = 'value' # value mean median
N_JOBS = -1
MIN_DAY = _C.MIN_DAY
MAX_DAY = _C.MAX_DAY
DEFAULT_DAYS_N = _C.DEFAULT_DAYS_N
FATS_MODE = 'all'

###################################################################################################################################################

def train_classifier(_train_df_x, train_df_y, _val_df_x, val_df_y, lcset_info,
	max_samples=None,
	nan_mode=NAN_MODE,
	):
	class_names = lcset_info['class_names']
	train_df_x, mean_train_df_x, null_cols = clean_df_nans(_train_df_x, mode=NAN_MODE)
	features = list(train_df_x.columns)
	best_rf = None
	best_rf_metric = -np.inf
	for criterion in ['gini', 'entropy']:
		for max_depth in [1, 2, 3, 4, 5][::-1]:
			rf = BalancedRandomForestClassifier( # BalancedRandomForestClassifier RandomForestClassifier
				n_jobs=N_JOBS,
				criterion=criterion,
				max_depth=max_depth,
				n_estimators=512, # 16 64 256 512
				max_samples=max_samples,
				max_features='auto', # None auto
				# min_samples_split=min_samples_split,
				bootstrap=True,
				#verbose=1,
				)
			rf.fit(train_df_x.values, train_df_y[['_y']].values[...,0])
			val_df_x, _, _ = clean_df_nans(_val_df_x, mode=NAN_MODE, df_values=mean_train_df_x)
			y_pred_p = rf.predict_proba(val_df_x.values)
			y_true = val_df_y[['_y']].values[...,0]
			metrics_cdict, metrics_dict, cm = fcm.get_multiclass_metrics(y_pred_p, y_true, class_names)
			rf_metric = metrics_dict['b-f1score'] # recall f1score
			recall = {c: metrics_cdict[c]['recall'] for c in class_names}
			print(f'train_samples={len(train_df_y)}; val_samples={len(val_df_y)}; features={len(features)}; criterion={criterion}; max_depth={max_depth}; rf_metric={rf_metric}; best_rf_metric={best_rf_metric}; recall={recall}')
			if rf_metric>best_rf_metric:
				best_rf = rf
				best_rf_metric = rf_metric
	
	### save best
	rank = TopRank('features', n=30)
	rank.add_list(features, best_rf.feature_importances_)
	rank.calcule()
	print(rank)
	d = {
		'rf':best_rf,
		'mean_train_df_x':mean_train_df_x,
		'null_cols':null_cols,
		'features':features,
		'rank':rank,
		}
	return d

###################################################################################################################################################

def evaluate_classifier(rf_d, fats_filedir, lcset_info,
	fats_mode=FATS_MODE,
	nan_mode=NAN_MODE,
	days_n=DEFAULT_DAYS_N,
	):
	class_names = lcset_info['class_names']
	features = rf_d['features']

	thdays_lengths = {}
	thdays_computed = []
	thdays_predictions = {}
	thdays_class_metrics_df = DFBuilder()
	thdays_class_metrics_cdf = {c:DFBuilder() for c in class_names}
	thdays_cm = {}
	thdays_class_metrics_all_bands_df = DFBuilder()
	thdays_class_metrics_all_bands_cdf = {c:DFBuilder() for c in class_names}

	thdays = np.linspace(MIN_DAY, MAX_DAY, days_n)
	for thday in thdays:
		eval_df_x, eval_df_y = load_features(fats_filedir,
			fats_mode=fats_mode,
			thday=thday,
			)
		lengths = eval_df_y[[c for c in list(eval_df_y.columns) if '_len' in c]].values # (n,b)
		thdays_lengths[thday] = np.sum(lengths, axis=-1) # (n,b)>(n)
		if np.all(np.any(lengths>=2, axis=-1)):
			y_true = eval_df_y[['_y']].values[...,0]
			eval_df_x, _, nan_cols = clean_df_nans(eval_df_x, mode=NAN_MODE, df_values=rf_d['mean_train_df_x'])
			y_pred_p = rf_d['rf'].predict_proba(eval_df_x.values)

			# metrics
			thdays_predictions[thday] = {'y_true':y_true, 'y_pred_p':y_pred_p}
			metrics_cdict, metrics_dict, cm = fcm.get_multiclass_metrics(y_pred_p, y_true, class_names)
			for c in class_names:
				thdays_class_metrics_cdf[c].append(None, update_dicts([{'_thday':thday}, metrics_cdict[c]]))
			thdays_class_metrics_df.append(None, update_dicts([{'_thday':thday}, metrics_dict]))

			### confusion matrix
			thdays_cm[thday] = cm

			### progress bar
			bmetrics_dict = {k:metrics_dict[k] for k in metrics_dict.keys() if 'b-' in k}
			print(f'thday={thday}; bmetrics_dict={bmetrics_dict}')
			thdays_computed += [thday]

			### all bands observed
			all_bands = np.all(lengths>=1, axis=-1) # (n,b)>(n)
			all_bands_y_pred_p = y_pred_p[all_bands]
			all_bands_y_true = y_true[all_bands]
			unique_classes = np.unique(all_bands_y_true)
			if len(unique_classes)==len(class_names):
				metrics_cdict, metrics_dict, cm = fcm.get_multiclass_metrics(all_bands_y_pred_p, all_bands_y_true, class_names)
				thdays_class_metrics_all_bands_df.append(thday, update_dicts([{'_thday':thday}, metrics_dict]))
				for c in class_names:
					thdays_class_metrics_all_bands_cdf[c].append(thday, update_dicts([{'_thday':thday}, metrics_cdict[c]]))

	d = {
		'model_name':f'mdl=brf',
		'survey':lcset_info['survey'],
		'band_names':lcset_info['band_names'],
		'class_names':class_names,
		'lcobj_names':list(eval_df_y[['_lcobj_name']].values[...,0]),

		'thdays':thdays,
		'thdays_lengths':thdays_lengths,
		'thdays_computed':thdays_computed,
		'thdays_predictions':thdays_predictions,
		'thdays_class_metrics_df':thdays_class_metrics_df.get_df(),
		'thdays_class_metrics_cdf':{c:thdays_class_metrics_cdf[c].get_df() for c in class_names},
		'thdays_cm':thdays_cm,
		'thdays_class_metrics_all_bands_df':thdays_class_metrics_all_bands_df.get_df(),
		'thdays_class_metrics_all_bands_cdf':{c:thdays_class_metrics_all_bands_cdf[c].get_df() for c in class_names},

		'features':features,
		'rank':rf_d['rank'],
		}
	return d