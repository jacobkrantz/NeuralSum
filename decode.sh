#!/bin/bash

# script cannot be called unless you first run:
#   >>> chmod u+x script_name.sh
# make these values the same across shell scripts:
USR_DIR=./NeuralSum
DATA_DIR=$USR_DIR/../data/tensor2tensor/data
TMP_DIR=$USR_DIR/../data/tensor2tensor/tmp
TRAIN_DIR=$USR_DIR/../data/tensor2tensor/train
PROBLEM=summary_problem_small
MODEL=my_custom_transformer
HPARAMS=exp_6
# location of file containing inputs to test against:
DECODE_FILE=./data/duc2003/sentences.txt
DECODE_FILE_OUT=./data/duc2003/generated.txt

# Example of populating DECODE_FILE with inputs to decode:
# echo "Makes vanish every star" >> $DECODE_FILE

BEAM_SIZE=2
ALPHA=0.6
TRAIN_STEPS=30000
EVAL_FREQ=1000
KEEP_CKPTS=20
WORKER_GPU=1

export CUDA_VISIBLE_DEVICES=1

# decode_hparams should include "extra_length"=14, but it does not work.
# we manually set the decode length to be 14 in my_custom_transformer.

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
