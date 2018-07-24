#!/bin/bash

# ------------------------------------------------------------------------------
# Filename:
#   t2t.sh
# Description:
#   entry point for data generation, training, and decoding for
#   tensor2tensor models.
# How to call this script:
#   >>> ./t2t.sh (datagen|train|decode)
# script cannot be called unless you first run:
#   >>> chmod +x script_name.sh
# ------------------------------------------------------------------------------

# Universal options:
USR_DIR=./NeuralSum
DATA_DIR=$USR_DIR/../data/tensor2tensor/data
TMP_DIR=$USR_DIR/../data/tensor2tensor/tmp
TRAIN_DIR=$USR_DIR/../data/tensor2tensor/train
PROBLEM=summary_problem
MODEL=my_custom_transformer
HPARAMS=exp_27
WORKER_GPU=1

# decoding-specific parameters:
# location of file containing inputs to test against:
DECODE_FILE=./data/duc2004/sentences.txt
DECODE_FILE_OUT=./data/duc2004/generated.txt
BEAM_SIZE=8
ALPHA=5.0

# training-specific parameters:
TRAIN_STEPS=25000
EVAL_FREQ=5000
KEEP_CKPTS=20

export CUDA_VISIBLE_DEVICES=1

# decode_hparams should include "extra_length"=14, but it does not work.
# we manually set the decode length to be 14 in my_custom_transformer.

use="$1"
if [ "$use" = "datagen" ]
then
  echo "Calling t2t-datagen..."
  t2t-datagen \
  	--t2t_usr_dir=$USR_DIR \
  	--data_dir=$DATA_DIR \
  	--tmp_dir=$TMP_DIR \
  	--problem=$PROBLEM
else
  if [ "$use" = "train" ]
  then
    echo "Calling t2t-trainer..."
    t2t-trainer \
    	--problem=$PROBLEM \
    	--model=$MODEL \
    	--hparams_set=$HPARAMS \
    	--data_dir=$DATA_DIR \
    	--output_dir=$TRAIN_DIR \
    	--t2t_usr_dir=$USR_DIR \
    	--train_steps=$TRAIN_STEPS \
    	--keep_checkpoint_max=$KEEP_CKPTS \
    	--local_eval_frequency=$EVAL_FREQ \
    	--worker_gpu=$WORKER_GPU
  else
    if [ "$use" = "decode" ]
    then
      echo "Calling t2t-decoder..."
      t2t-decoder \
        --data_dir=$DATA_DIR \
        --problem=$PROBLEM \
        --model=$MODEL \
        --hparams_set=$HPARAMS \
        --output_dir=$TRAIN_DIR \
        --t2t_usr_dir=$USR_DIR \
        --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA","force_decode_length"=True \
        --decode_from_file=$DECODE_FILE \
        --decode_to_file=$DECODE_FILE_OUT \
        --stop_at_eos=True \
        --decode_interactive=False \
        --worker_gpu=$WORKER_GPU
    else
      echo "Arg 1 must be one of [datagen, train, or decode]."
      exit 127
    fi
  fi
fi
