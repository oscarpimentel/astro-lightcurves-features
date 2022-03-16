from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import pandas as pd
from fuzzytools.progress_bars import ProgressBar
import lc_classifier.features as lccf
from fuzzytools.multiprocessing import get_joblib_config_batches
from joblib import Parallel, delayed
from fuzzytools.dataframes import DFBuilder
from copy import copy, deepcopy

MIN_DAY = _C.MIN_DAY
MAX_DAY = 100
DEFAULT_DAYS_N = _C.DEFAULT_DAYS_N
USES_MAGNITUDE = True

###################################################################################################################################################

def preprocess_lcobj(_lcobj, band_names, thday,
	uses_magnitude=USES_MAGNITUDE,
	uses_band_d=False,
	):
	lcobj = copy(_lcobj)
	if uses_magnitude:
		lcobj.convert_to_magnitude()

	lightcurve = DFBuilder()
	band_d = {'g':1, 'r':2}
	lengths = {}

	lcobj.reset_day_offset_serial() # remove day offset!
	lcobj.clip_attrs_given_max_day(thday) # clip by max day

	for b in band_names:
		lcobjb = lcobj.get_b(b)
		lengths[f'_len_{b}'] = len(lcobjb)
		for k in range(0, len(lcobjb)):
			lightcurve.append(f'{b}.{k}', {
				'oid':'',
				'time':lcobjb.days[k],
				'magpsf':lcobjb.obs[k],
				'magnitude':lcobjb.obs[k],
				'sigmapsf':lcobjb.obse[k],
				'error':lcobjb.obse[k],
				'band':band_d[b] if uses_band_d else b,
				'isdiffpos':1,
				})

	lightcurve = lightcurve.get_df().set_index('oid')
	return lightcurve, lengths

###################################################################################################################################################

def get_features(lcobj, lcobj_name, lcset_name, lcset_info,
	days_n=DEFAULT_DAYS_N,	
	):
	band_names = lcset_info['band_names']
	thdays = np.linspace(MIN_DAY, MAX_DAY, days_n) if 'test' in lcset_name else [MAX_DAY]
	thdays_features_list = []
	for thday in thdays:
		thday_features = {
			'_lcobj_name':lcobj_name,
			'_thday':thday,
			'_y':lcobj.y,
			'_fullsynth':lcobj.all_synthetic(),
			}
		lightcurve, lengths = preprocess_lcobj(lcobj, band_names, thday, uses_band_d=True)
		thday_features.update(lengths)
		features_d = lccf.FeatureExtractorComposer([lccf.ZTFColorFeatureExtractor()]).compute_features(lightcurve).to_dict(orient='index')['']
		thday_features.update(features_d)

		feature_extractor = lccf.FeatureExtractorComposer([
			# lccf.SGScoreExtractor(), # metadata?
			# lccf.SupernovaeDetectionAndNonDetectionFeatureExtractor(band_names), # metadata?
			lccf.SupernovaeDetectionFeatureExtractor(band_names),
			lccf.SNParametricModelExtractor(band_names),
			lccf.IQRExtractor(band_names),
			lccf.MHPSExtractor(band_names),
			lccf.TurboFatsFeatureExtractor(band_names),
			lccf.GPDRWExtractor(band_names),
			lccf.PowerRateExtractor(band_names), # for periodic object
			lccf.PeriodExtractor(band_names), # for periodic object
			lccf.HarmonicsExtractor(band_names), # for periodic object

			# lccf.FoldedKimExtractor(band_names),
			# lccf.ZTFForcedPhotometryFeatureExtractor(band_names),
			# lccf.ZTFLightcurvePreprocessor(band_names),
			# lccf.SNFeaturesPhaseIIExtractor(band_names),
			# lccf.SPMExtractorPhaseII(band_names),
			# lccf.ZTFFeatureExtractor(band_names), # slow
			# lccf.GalacticCoordinatesExtractor(band_names),
			# lccf.RealBogusExtractor(band_names),
			# lccf.StreamSGScoreExtractor(band_names),
			# lccf.WiseStaticExtractor(band_names),
			# lccf.WiseStreamExtractor(band_names),
			])
		lightcurve, lengths = preprocess_lcobj(lcobj, band_names, thday)
		features_d = feature_extractor.compute_features(lightcurve).to_dict(orient='index')['']
		thday_features.update(features_d)
		thdays_features_list += [thday_features]
	return thdays_features_list

def get_all_fat_features(lcdataset, lcset_name,
	backend=None, # None multiprocessing
	):
	lcset = lcdataset[lcset_name]
	band_names = lcset.band_names
	thdays_features_df = DFBuilder()
	lcobj_names = lcset.get_lcobj_names()
	batches, n_jobs = get_joblib_config_batches(lcobj_names, backend)
	bar = ProgressBar(len(batches))
	for batch in batches:
		bar(f'lcset_name={lcset_name}; batch={batch}({len(batch)}#)')
		jobs = []
		for lcobj_name in batch:
			jobs.append(delayed(get_features)(
				lcset[lcobj_name],
				lcobj_name,
				lcset_name,
				lcset.get_info(),
			))
		results = Parallel(n_jobs=n_jobs, backend=backend)(jobs)
		for thdays_features_list in results:
			for thdays_features in thdays_features_list:
				thdays_features_df.append(None, thdays_features)

	bar.done()
	return thdays_features_df.get_df()
