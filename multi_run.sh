#!/bin/bash
# ------------------------------------------------------------------------------
# Script for running multiple experiments in one shot.
# 	Saves score reports to ./data/tensor2tensor/saved_models/auto/
# Call once to make script runnable (user permission. +x grants all):
# 	>>> chmod u+x script_name.sh
# t2t-trainer.sh and decode.sh needs line HPARAMS=exp_6
# ------------------------------------------------------------------------------


# Generate and preprocess data
./t2t.sh datagen && python process_duc.py 2004

# Make folder to store generated models
mkdir ./data/tensor2tensor/saved_models/auto

old=6
for experiment in 27 28 29 30 31 32 33 34 35 36
do
	echo "Running experiment $experiment..."

	# Update parameter sets in t2t.sh run script
	str='s/exp_'$old'/exp_'$experiment'/g;'
	perl -i -p -e $str ./t2t.sh
	old=$experiment

	# Train, decode, and evaluate experiment model
	./t2t.sh train
	./t2t.sh decode
	python evaluate_on_duc.py --which_duc 2004 --save_report True

	# Move model files out of training folder to make room for next run
	mkdir ./data/tensor2tensor/saved_models/auto/exp_$experiment
	mv ./data/tensor2tensor/train/* ./data/tensor2tensor/saved_models/auto/exp_$experiment/

	# Rename output files to experiment-specific name
	mv ./data/out/vert_reports/vert_scores* ./data/tensor2tensor/saved_models/auto/vert_exp_$experiment.json
	mv ./data/duc2004/generated.txt ./data/tensor2tensor/saved_models/auto/generated_$experiment.txt
done
echo "All experiments done."
