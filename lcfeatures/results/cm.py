from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import fuzzytools.files as ftfiles
import fuzzytools.strings as ftstrings
from fuzzytools.datascience.cms import ConfusionMatrix
from fuzzytools.matplotlib.cm_plots import plot_custom_confusion_matrix
import matplotlib.pyplot as plt
from fuzzytools.datascience.xerror import XError
from IPython.display import display
from fuzzytools.strings import latex_bf_alphabet_count
from fuzzytools.latex.latex_tables import LatexTable
from fuzzytools.matplotlib.utils import save_fig
import fuzzytools.strings as strings
import lcfeatures.results.utils as utils

FIGSIZE = (6,5)
DPI = 200
RANDOM_STATE = None
NEW_ORDER_CLASS_NAMES = ['SNIa', 'SNIbc', 'SNII*', 'SLSN']
DICT_NAME = 'thdays_class_metrics'

###################################################################################################################################################

def plot_cm(rootdir, cfilename, kf, lcset_name, model_names,
	figsize=FIGSIZE,
	dpi=DPI,
	new_order_class_names=NEW_ORDER_CLASS_NAMES,
	dict_name=DICT_NAME,
	alphabet_count=0,
	verbose=0,
	):
	for model_name in model_names:
		fmodel_name, mn_dict = utils.get_fmodel_name(model_name, returns_mn_dict=True)
		method = mn_dict['method']
		load_roodir = f'../save/{model_name}/performance/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method={method}'
		print(load_roodir)
		files, files_ids, kfs = ftfiles.gather_files_by_kfold(load_roodir, kf, lcset_name,
			fext='d',
			imbalanced_kf_mode='oversampling', # error oversampling
			random_state=RANDOM_STATE,
			)
		
		print(f'{files_ids}({len(files_ids)}#)')
		if len(files)==0:
			continue

		class_names = files[0]()['class_names']
		features = files[0]()['features']
		thdays = files[0]()['thdays']
		rank = files[0]()['rank']
		for f in features:
			#print(f)
			pass
		thday = files[0]()['thdays'][-1]
		xe_dict = {}
		for metric_name in ['recall', 'f1score']:
			xe_metric = XError([f()['thdays_class_metrics_df'].loc[f()['thdays_class_metrics_df']['_thday']==thday][f'b-{metric_name}'].item() for f in files])
			xe_dict[f'b-{metric_name}'] = xe_metric

		brecall_xe = xe_dict['b-recall']
		bf1score_xe = xe_dict['b-f1score']

		new_order_class_names = ['SNIa', 'SNIbc', 'SNIIbn', 'SLSN']
		new_order_class_names = ['SNIa', 'SNIbc', 'SNII*', 'SLSN']
		cm = ConfusionMatrix([f()['thdays_cm'][thday] for f in files], class_names)
		cm.reorder_classes(new_order_class_names)
		for c in new_order_class_names:
			print(cm.get_diagonal_dict()[c].get_raw_repr(f'brf_{c}_tp'))
			pass
		true_label_d = {c:f'({k}#)' for c,k in zip(class_names, np.sum(files[0]()['thdays_cm'][thday], axis=1))}

		rank = files[0]()['rank'] # just show one
		rank.names = ['Feature name=\\verb+'+n+'+' for n in rank.names]
		rank.values = [v*100 for v in rank.values]
		rank_df = rank.get_df()
		latex_table = LatexTable(rank_df,
			label='tab:brf_ranking',
			)
		if verbose:
			display(rank_df)
			print(latex_table)

		title = ''
		title += f'{latex_bf_alphabet_count(alphabet_count)}{fmodel_name}'+'\n'
		title += f'b-Recall={brecall_xe}; b-$F_1$score={bf1score_xe}'+'\n'
		title += f'th-day={thday:.0f} [days]'+'\n'
		fig, ax = plot_custom_confusion_matrix(cm,
			title=title[:-1],
			figsize=figsize,
			dpi=dpi,
			true_label_d=true_label_d,
			lambda_c=lambda c:c.replace('*', ''),
			)
		save_fig(fig, f'../temp/exp=cm/{model_name}.pdf', closes_fig=0)
		plt.show()