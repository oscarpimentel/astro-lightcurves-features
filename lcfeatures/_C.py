import sys
import numpy as np

###################################################################################################################################################

MIN_DAY = 1
MAX_DAY = 100
DEFAULT_DAYS_N = int(MAX_DAY-MIN_DAY+1)

SNE_SELECTED_FEATURES = [
	'SPM_t0', # *
	'SPM_A', # *
	'SPM_chi', # *
	'SPM_gamma', # *
	'SPM_tau_rise', # *
	'SPM_tau_fall', # *
	'SPM_beta', # *

	'LinearTrend', # *
	'IAR_phi', # *
	'MHPS_ratio', # *
	'GP_DRW_tau', # * slow
	'GP_DRW_sigma', # slow

	'MHPS_low', # *
	'MHPS_PN_flag',
	'MHPS_high',
	'MHPS_non_zero',

	'SF_ML_amplitude', # *
	]

HARMONICS_FEATURES = [
	'Harmonics_mag_2_1',
	'Harmonics_mag_3_1',
	'Harmonics_mag_4_1',
	'Harmonics_mag_5_1',
	'Harmonics_mag_6_1',
	'Harmonics_mag_7_1',
	'Harmonics_mse_1',
	'Harmonics_phase_2_1',
	'Harmonics_phase_3_1',
	'Harmonics_phase_4_1',
	'Harmonics_phase_5_1',
	'Harmonics_phase_6_1',
	'Harmonics_phase_7_1',
	]

NOT_IMPLEMENTED = [
	'median_diffmaglim_before_fid',
	'Period_band',
	'mean_mag',
	'delta_period',
	'n_pos',
	'min_mag',
	'first_mag',
	'positive_fraction',
	'Psi_CS',
	'Psi_eta',
	'dmag_non_det_fid',
	'MeanvariancePairSlopeTrend',
	'delta_mag_fid',
	'delta_mjd_fid',
	'dmag_first_det_fid',
	'iqr',
	'last_diffmaglim_before_fid',
	'last_mjd_before_fid',
	'max_diffmaglim_after_fid',
	'max_diffmaglim_before_fid',
	'n_det',
	'n_neg',
	'n_non_det_after_fid',
	'n_non_det_before_fid',
	'VariabilityIndex',
	]

###################################################################################################################################################

ALERCE_SPM_FEATURES = [
	'SPM_t0',
	'SPM_A',
	'SPM_gamma',
	'SPM_tau_rise',
	'SPM_tau_fall',
	'SPM_beta',
	'SPM_chi',
	]

MPHS_FEATURES = [
	'MHPS_PN_flag',
	'MHPS_high',
	'MHPS_low',
	'MHPS_non_zero',
	'MHPS_ratio',
	]

TURBOFATS_FEATURES = [
	'PeriodLS_v2', # slow
	'Period_fit_v2', # slow
	'Amplitude',
	'AndersonDarling',
	'Autocor_length',
	'Beyond1Std',
	'Con',
	'Eta_e',
	'ExcessVar',
	'GP_DRW_sigma', # slow
	'GP_DRW_tau', # slow
	'Gskew',
	'Harmonics', # slow
	'IAR_phi',
	'LinearTrend',
	'MaxSlope',
	'Mean',
	'Meanvariance',
	'MedianAbsDev',
	'MedianBRP',
	'PairSlopeTrend',
	'PercentAmplitude',
	'Pvar',
	'Q31',
	'Rcs',
	'SF_ML_amplitude',
	'SF_ML_gamma',
	'Skew',
	'SmallKurtosis',
	'Std',
	'StetsonK',
	'CAR_sigma',
	'CAR_mean',
	'CAR_tau',
	'FluxPercentileRatioMid20',
	'FluxPercentileRatioMid35',
	'FluxPercentileRatioMid50',
	'FluxPercentileRatioMid65',
	'FluxPercentileRatioMid80',
	'PercentDifferenceFluxPercentile',
	]