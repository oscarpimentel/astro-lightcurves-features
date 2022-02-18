#!/bin/bash
clear
SECONDS=0
run_script(){
	echo "$1"; eval "$1";
}

###################################################################################################################################################
methods=(
	spm-mcmc-estw
	spm-mcmc-fstw
	linear-fstw
	bspline-fstw
	)

for method in "${methods[@]}"; do
	for kf in {0..4}; do # 0..4
		# run_script "python generate_fats_features.py --method $method --kf $kf --ignore_train 0 --ignore_synth 0"
		:
	done
done

for mid in {1000..1005}; do # 1000..1005
	for method in "${methods[@]}"; do
		for kf in {0..4}; do # 0..4
			run_script "python train_rf_models.py --method $method --kf $kf --mid $mid"
			# run_script "python export_2dprojections.py --method $method --kf $kf --mid $mid"
			:
		done
	done
done

###################################################################################################################################################
mins=$((SECONDS/60))
echo echo "time elapsed=${mins} [mins]"