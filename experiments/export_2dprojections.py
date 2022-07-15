#!/usr/bin/env python3
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
parser.add_argument('--mid',  type=str)
main_args = parser.parse_args()
print_big_bar()

###################################################################################################################################################
import numpy as np
from fuzzytools.files import load_pickle, save_pickle, get_dict_from_filedir

filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method={main_args.method}.splcds'
filedict = get_dict_from_filedir(filedir)
rootdir = filedict['_rootdir']
cfilename = filedict['_cfilename']
lcdataset = load_pickle(filedir)
lcset_info = lcdataset['raw'].get_info()
lcdataset.only_keep_kf(main_args.kf) # saves ram
# print(lcdataset)

###################################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from lcfeatures.projs import get_fitted_2dproj
from lcfeatures.plots.projs import plot_2dproj
from fuzzytools.progress_bars import ProgressBar
from fuzzytools.matplotlib.utils import save_fig
from fuzzytools.strings import get_string_from_dict

lcset_name = f'{main_args.kf}@train.{main_args.method}'
load_rootdir = f'../save/fats/{cfilename}'

for proj_mode,features_mode in [('UMAP', 'all'), ('UMAP', 'spm'), ('PCA', 'spm')]:
        proj_dict = get_fitted_2dproj(lcdataset, lcset_name, load_rootdir, proj_mode, features_mode)
        class_names = lcdataset[lcset_name].class_names
        for target_class in [None]+class_names:
                fig = plot_2dproj(proj_dict, lcdataset, lcset_name, target_class)
                train_mode = 'r+s'
                save_rootdir = f'../save'
                if target_class is None:
                        save_filedir = f'{save_rootdir}/method={main_args.method}~tmode={train_mode}~fmode={features_mode}/projections/{cfilename}/{lcset_name}/proj_mode={proj_mode}/id={main_args.mid}.pdf'
                else:
                        save_filedir = f'{save_rootdir}/method={main_args.method}~tmode={train_mode}~fmode={features_mode}/projections/{cfilename}/{lcset_name}/proj_mode={proj_mode}/{target_class}/id={main_args.mid}.pdf'
                
                fig.tight_layout()
                save_fig(fig, save_filedir)
