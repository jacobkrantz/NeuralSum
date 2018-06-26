#!/bin/bash

# script cannot be called unless you first run:
#   >>> chmod u+x script_name.sh
# make these values the same across shell scripts:
USR_DIR=./NeuralSum
DATA_DIR=$USR_DIR/../data/tensor2tensor/data
TMP_DIR=$USR_DIR/../data/tensor2tensor/tmp
TRAIN_DIR=$USR_DIR/../data/tensor2tensor/train
PROBLEM=summary_problem
MODEL=my_custom_transformer
HPARAMS=exp_6
# location of file containing inputs to test against:
DECODE_FILE=./data/duc2004/sentences.txt
DECODE_FILE_OUT=./data/duc2004/generated.txt

export CUDA_VISIBLE_DEVICES=1

t2t-trainer \
	--problem=$PROBLEM \
	--model=$MODEL \
	--hparams_set=$HPARAMS \
	--data_dir=$DATA_DIR \
	--output_dir=$TRAIN_DIR \
	--t2t_usr_dir=$USR_DIR \
	--train_steps=200000 \
	--keep_checkpoint_max=20 \
	--local_eval_frequency=10000 \
	--worker_gpu=1
