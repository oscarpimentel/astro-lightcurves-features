from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from fuzzytools.matplotlib.colors import get_scaled_color
from fuzzytools.datascience.ranks import TopRank
from lchandler._C import CLASSES_STYLES
from fuzzytools.strings import latex_bf_alphabet_count

FIGSIZE = (8,8)
DPI = 200

###################################################################################################################################################

def get_synth_objs(obj_r, objs_s,
	max_synth_samples=1,
	):
	objs = []
	counter = 0
	for k,obj_s in enumerate(objs_s):
		name_r = obj_r
		name_s,id_s = obj_s.split('.')
		if name_r==name_s:
			objs.append(obj_s)
			counter += 1
			if counter>=max_synth_samples:
				break
	return objs

###################################################################################################################################################

def plot_2dproj(proj_dict, lcdataset, s_lcset_name, target_class,
	r_max_samples=np.inf,
	figsize=FIGSIZE,
	dpi=DPI,
	):
	r_lcset = lcdataset[s_lcset_name.split('.')[0]]
	fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
	class_names = proj_dict['class_names']
	rank_names = []
	for kc,c in enumerate(class_names):
		ax = axs
		p_kwargs = {
			'r_max_samples':r_max_samples,
			}
		if target_class is None:
			rank_names += plot_2dproj_c(ax, proj_dict, c, True, False, **p_kwargs)
		else:
			rank_names += plot_2dproj_c(ax, proj_dict, c, c==target_class, True, **p_kwargs)

	title = ''
	if target_class is None:
		title += f'reduction method={proj_dict["proj_mode"]}'+'\n'
	else:
		title += f'{latex_bf_alphabet_count(class_names.index(target_class))} reduction_method={proj_dict["proj_mode"]}; class={target_class.replace("*", "")}'+'\n'
	
	if len(rank_names)>0:
		title += f'real-synth top distance objs=[{", ".join(rank_names)}]'+'\n'  
	ax.legend()
	ax.set_title(title[:-1])
	ax.grid(alpha=0.0)
	return fig

def plot_2dproj_c(ax, proj_dict, c, is_target_class, plots_net,
	r_max_samples=np.inf,
	s_max_samples=1,
	rank=3,
	):
	map_lcobj_names = proj_dict['map_lcobj_names']
	class_names = proj_dict['class_names']
	map_x = proj_dict['map_x']
	labels = proj_dict['y']
	r_counter = 0
	dist_rank = TopRank()

	### plot all
	for idx,lcobj_name in enumerate(map_lcobj_names):
		if '.' in lcobj_name: # is synthetic
			continue
		if r_counter>r_max_samples:
			continue
		if not class_names[labels[idx]]==c:
			continue

		scatter_kwargs = {
			's':32*CLASSES_STYLES[c]['markerprop'],
			'color':get_scaled_color(CLASSES_STYLES[c]['c'], 2.5) if c=='SNIa' else CLASSES_STYLES[c]['c'],
			'marker':CLASSES_STYLES[c]['marker'],
			'linewidth':0,
			'zorder':CLASSES_STYLES[c]['zorder'],
			'label':f'{c.replace("*", "")} [real]' if r_counter==0 else None,
			}
		map_x_real = map_x[idx]
		if not is_target_class:
			scatter_kwargs['color'] = 'k'
			scatter_kwargs['zorder'] = -100
		plt.scatter(map_x_real[0], map_x_real[1], **scatter_kwargs)

		### plot synthetics
		lcobj_names_synth = get_synth_objs(lcobj_name, proj_dict['s_lcobj_names'], s_max_samples)
		for ks,lcobj_name_synth in enumerate(lcobj_names_synth):
			idx = map_lcobj_names.index(lcobj_name_synth)
			map_x_synth = map_x[idx]
			scatter_kwargs = {
				's':30*CLASSES_STYLES[c]['markerprop'],
				'color':get_scaled_color(CLASSES_STYLES[c]['c'], 2.5) if c=='SNIa' else CLASSES_STYLES[c]['c'],
				'marker':'*',
				'linewidth':0,
				'zorder':CLASSES_STYLES[c]['zorder'],
				'label':f'{c.replace("*", "")} [synth]' if r_counter==0 and ks==0 else None,
				}
			if not is_target_class:
				scatter_kwargs['color'] = 'k'
				scatter_kwargs['zorder'] = -100
			ax.scatter(map_x_synth[0], map_x_synth[1], **scatter_kwargs)

			if is_target_class and plots_net:
				dx = map_x_real[0]-map_x_synth[0]
				dy = map_x_real[1]-map_x_synth[1]
				dist = dx**2+dy**2
				line_kwargs = {
					'alpha':0.4,
					'lw':0.3,
					'c':CLASSES_STYLES[c]['c'],
					'linestyle':'--',
					}
				line = ax.plot([map_x_real[0], map_x_synth[0]], [map_x_real[1], map_x_synth[1]], **line_kwargs)
				line_kwargs['alpha'] = 1
				line_kwargs['linestyle'] = '-'
				ax.plot([None], [None], label=f'{c.replace("*", "")} real-synth' if r_counter==0 and ks==0  else None, **line_kwargs)
				info = {
					# 'pos':(map_x_real[0], map_x_real[1]),
					# 'pos':(map_x_synth[0], map_x_synth[1]),
					'pos':((map_x_real[0]+map_x_synth[0])/2, (map_x_real[1]+map_x_synth[1])/2),
					'line':line[0],
					}
				dist_rank.append(lcobj_name_synth, dist, info)

		r_counter += 1

	### rank
	rank_names = []
	if is_target_class and plots_net:
		dist_rank.calcule()
		for k in range(0, rank):
			name, value, info = dist_rank[k]
			rank_names += [name]
			info['line'].set_alpha(1)
			info['line'].set_linestyle('-')
			dist_txt = f'{value:.1f}'
			# txt = f'{name}; {k}; {dist_txt}'
			txt = f'{dist_txt}'
			txt = ax.text(*info['pos'], txt,
				horizontalalignment='center',
				verticalalignment='center', # center top bottom baseline
				fontsize=7,
				c=CLASSES_STYLES[c]['c'],
				)
			txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

	return rank_names