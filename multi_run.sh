#!/bin/bash
# ------------------------------------------------------------------------------
# Script for running multiple experiments in one shot.
# 	Saves score reports to ./data/out/vert_scores/
# Call once to make script runnable (user permission. +x grants all):
# 	>>> chmod u+x script_name.sh
# t2t-trainer.sh and decode.sh needs line HPARAMS=exp_13
# ------------------------------------------------------------------------------


# Generate and preprocess data
./datagen.sh && python process_duc.py 2003

old=13
for experiment in 11 12 14 15 16 17 18
do
	echo "Running experiment $experiment..."

	# update parameter sets in t2t-trainer.sh and decode.sh
	str='s/exp_'$old'/exp_'$experiment'/g;'
	perl -i -p -e $str ./t2t-trainer.sh
	perl -i -p -e $str ./decode.sh
	old=$experiment

	# train, decode, and evaluate experiment model
	./t2t-trainer.sh
	./decode.sh
	python evaluate_on_duc.py save

	# move model files out of training folder to make room for next run
	mkdir ./data/tensor2tensor/old_models/exp_$experiment
	mv ./data/tensor2tensor/train/* ./data/tensor2tensor/old_models/exp_$experiment

	# rename vert score to experiment specific name
	mv ./data/out/vert_reports/vert_scores* ./vert_exp_$experiment.json
done
echo "All experiments done."
