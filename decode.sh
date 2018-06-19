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
HPARAMS=my_custom_hparams
# location of file containing inputs to test against:
DECODE_FILE=./data/duc2004/sentences.txt
DECODE_FILE_OUT=./data/duc2004/generated.txt

# Example of populating DECODE_FILE with inputs to decode:
# echo "Makes vanish every star" >> $DECODE_FILE

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$USR_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA","force_decode_length"=True,"extra_length"=12 \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=$DECODE_FILE_OUT \
  --stop_at_eos=True \
  --decode_interactive=False
  --worker_gpu=1
