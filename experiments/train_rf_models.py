#!/usr/bin/env python3
# -*- coding: utf-8 -*
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../fuzzy-tools') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module
sys.path.append('../../astro-lightcurves-fats') # or just install the module

###################################################################################################################################################
import argparse
from fuzzytools.prints import print_big_bar

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--method',  type=str)
parser.add_argument('--kf',  type=str)
parser.add_argument('--mid',  type=str, default='0')
parser.add_argument('--classifier_mids',  type=int, default=2)
main_args = parser.parse_args()
print_big_bar()

###################################################################################################################################################
import numpy as np
from fuzzytools.files import load_pickle, save_pickle, get_dict_from_filedir
from lcfeatures.files import load_features
from fuzzytools.progress_bars import ProgressBar
from lcfeatures.classifiers import train_classifier, evaluate_classifier
import pandas as pd

filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method={main_args.method}.splcds'
filedict = get_dict_from_filedir(filedir)
rootdir = filedict['_rootdir']
cfilename = filedict['_cfilename']
lcdataset = load_pickle(filedir)
lcset_info = lcdataset['raw'].get_info()
lcdataset.only_keep_kf(main_args.kf) # saves ram
# print(lcdataset)

train_modes = ['r', 's', 'r+s'] if main_args.method=='spm-mcmc-estw' else ['s', 'r+s']
for train_mode in train_modes:
	for classifier_mid in range(0, main_args.classifier_mids):
		print(f'training brf for train_mode={train_mode}; kf={main_args.kf}; method={main_args.method}; mid={main_args.mid}c{classifier_mid}')
		train_df_x_r, train_df_y_r = load_features(f'../save/fats/{cfilename}/{main_args.kf}@train.df')
		if train_mode=='r':
			train_df_x = pd.concat([train_df_x_r], axis='rows')
			train_df_y = pd.concat([train_df_y_r], axis='rows')

		if train_mode=='s':
			train_df_x_s, train_df_y_s = load_features(f'../save/fats/{cfilename}/{main_args.kf}@train.{main_args.method}.df')
			train_df_x = pd.concat([train_df_x_s], axis='rows')
			train_df_y = pd.concat([train_df_y_s], axis='rows')

		if train_mode=='r+s':
			train_df_x_s, train_df_y_s = load_features(f'../save/fats/{cfilename}/{main_args.kf}@train.{main_args.method}.df')
			s_repeats = len(train_df_x_s)//len(train_df_x_r)
			train_df_x = pd.concat([train_df_x_r]*s_repeats+[train_df_x_s], axis='rows')
			train_df_y = pd.concat([train_df_y_r]*s_repeats+[train_df_y_s], axis='rows')

		val_df_x, val_df_y = load_features(f'../save/fats/{cfilename}/{main_args.kf}@val.df')
		brf_d = train_classifier(train_df_x, train_df_y, val_df_x, val_df_y, lcset_info,
			max_samples=len(train_df_x_r),
			)
		d = evaluate_classifier(brf_d, f'../save/fats/{cfilename}/{main_args.kf}@test.df', lcset_info)
		features_mode = 'all'
		save_rootdir = f'../save'
		save_filedir = f'{save_rootdir}/method={main_args.method}~tmode={train_mode}~fmode={features_mode}/performance/{cfilename}/{main_args.kf}@test/id={main_args.mid}c{classifier_mid}.d'
		save_pickle(save_filedir, d)