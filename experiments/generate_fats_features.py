#!/usr/bin/env python3
# -*- coding: utf-8 -*
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../fuzzy-tools') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module

###################################################################################################################################################
import argparse
from fuzzytools.prints import print_big_bar

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--method',  type=str)
parser.add_argument('--kf',  type=str)
parser.add_argument('--ignore_train', type=int, default=0)
parser.add_argument('--ignore_synth', type=int, default=0)
main_args = parser.parse_args()
print_big_bar()

###################################################################################################################################################
import numpy as np
from fuzzytools.files import load_pickle, save_pickle
from fuzzytools.files import get_dict_from_filedir
from lcfeatures.extractors import get_all_fat_features
from lcfeatures.files import save_features

filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method={main_args.method}.splcds'
filedict = get_dict_from_filedir(filedir)
rootdir = filedict['_rootdir']
cfilename = filedict['_cfilename']
lcdataset = load_pickle(filedir)
lcset_info = lcdataset['raw'].get_info()
lcdataset.only_keep_kf(main_args.kf) # saves ram
print(lcdataset)

lcset_names = lcdataset.get_lcset_names()
for lcset_name in lcset_names:
	if main_args.ignore_train and 'train' in lcset_name:
		continue
	if main_args.ignore_synth and '.' in lcset_name:
		continue
	is_kf = '@' in lcset_name and lcset_name.split('@')[0]==main_args.kf
	if len(lcdataset[lcset_name])>0 and is_kf:
		thdays_features_df = get_all_fat_features(lcdataset, lcset_name)
		save_rootdir = f'../save'
		save_filedir = f'{save_rootdir}/fats/{cfilename}/{lcset_name}.df'
		save_features(thdays_features_df, save_filedir)
	else:
		pass


