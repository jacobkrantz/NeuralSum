#!/bin/bash

# script cannot be called unless you first run:
#   >>> chmod u+x script_name.sh
# make these values the same across shell scripts:
USR_DIR=./NeuralSum
DATA_DIR=$USR_DIR/../data/tensor2tensor/data
TMP_DIR=$USR_DIR/../data/tensor2tensor/tmp
TRAIN_DIR=$USR_DIR/../data/tensor2tensor/train
PROBLEM=summary_problem
MODEL=transformer
HPARAMS=my_custom_hparams
# location of file containing inputs to test against:
DECODE_FILE=./data/duc2004/sentences.txt
DECODE_FILE_OUT=./data/duc2004/generated.txt

t2t-datagen \
	--t2t_usr_dir=$USR_DIR \
	--data_dir=$DATA_DIR \
	--tmp_dir=$TMP_DIR \
	--problem=$PROBLEM
