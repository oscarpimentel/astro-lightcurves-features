from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
from fuzzytools.files import load_pickle, save_pickle
from fuzzytools.datascience.xerror import XError
import matplotlib.pyplot as plt
import fuzzytools.strings as ftstrings
import fuzzytools.files as ftfiles
import fuzzytools.matplotlib.fills as fills
import lcfeatures.results.utils as utils
from fuzzytools.matplotlib.utils import save_fig

RANDOM_STATE = None
STD_PROP = 1
SHADOW_ALPHA = .1
FIGSIZE = (8, 5)
DPI = 200
METRICS_D = _C.METRICS_D
FILL_USES_MEAN = True
DICT_NAME = 'thdays_class_metrics'
N_XTICKS = 4

###################################################################################################################################################

def plot_metric(rootdir, cfilename, kf, set_name, model_names, metric_names,
	target_class=None,
	figsize=FIGSIZE,
	dpi=DPI,
	std_prop=STD_PROP,
	shadow_alpha=SHADOW_ALPHA,
	dict_name=DICT_NAME,
	n_xticks=N_XTICKS,
	):
	for metric_name in metric_names:
		new_metric_name = 'b-'+METRICS_D[metric_name]['mn'] if target_class is None else METRICS_D[metric_name]['mn']+'$_\\mathregular{('+target_class+')}$'
		fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		for kmn,model_name in enumerate(model_names):
			fmodel_name, mn_dict = utils.get_fmodel_name(model_name, returns_mn_dict=True)
			method = mn_dict['method']
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
			thdays_computed_curves = []
			metric_curves = []
			for f in files:
				thdays_computed_curves += [np.array(f()['thdays_computed'])]
				if target_class is None:
					metric_curve = f()[f'{dict_name}_df'][f'b-{metric_name}'].values
					metric_curves += [metric_curve]
				else:
					metric_curve = f()[f'{dict_name}_cdf'][target_class][f'{metric_name}'].values
					metric_curves += [metric_curve]
			
			color = utils.get_model_color(model_name, model_names)
			label = f'{fmodel_name}'
			shadow_alpha = 0.1
			std_prop = .5
			train_mode = mn_dict['tmode']
			if train_mode=='r':
				linestyle = '-'
				color = 'k'
			elif train_mode=='s':
				linestyle = ':'
			elif train_mode=='r+s':
				linestyle = '-.'
			else:
				raise Exception(f'train_mode={train_mode}')
			ax, new_x, median_y, yrange = fills.fill_beetween_mean_std(ax, thdays_computed_curves, metric_curves,
				mean_kwargs={'color':color, 'alpha':1, 'linestyle':linestyle, 'marker':'D', 'markersize':0, 'markerfacecolor':'None', 'markevery':[0], 'zorder':-.5, 'label':label},
				fill_kwargs={'color':color, 'alpha':shadow_alpha, 'lw':0, 'zorder':-.5},
				returns_extras=True,
				std_prop=std_prop,
				)
			if kmn==len(model_names)-1:
				ax.axvline(new_x[0],
					linestyle='--',
					c='k',
					label=f'threshold-day={new_x[0]:.0f} [days]',
					zorder=-1,
					lw=1,
					)
			ax.set_xticks(thdays[::-1][::n_xticks][::-1])
			ax.set_xticklabels([f'{xt:.0f}' for xt in ax.get_xticks()], rotation=90)
			ax.set_xlabel('threshold-day [days]')
			ax.set_ylabel(new_metric_name)

			title = ''
			title += f'{new_metric_name} curve using the moving threshold-day'+'\n'
			ax.set_title(title[:-1])
			ax.set_xlim([min(thdays), max(thdays)])
			ax.legend(loc='lower right')

		fig.tight_layout()
		save_fig(fig, f'../temp/exp=performance_curve~metric_name={metric_name}.pdf', closes_fig=0)
		plt.show()