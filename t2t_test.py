# Imports we need.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import collections

# must import problem class to use external problem
from NeuralSum import summary_problem

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
"""
Words are represented as numbers. They are essentially one hot encoded
    because these numbers perform a lookup in the vocabulary. Look at
    the voab file.
"""
def largest_input_tensor_size(dataset):
    return max([int(ex['inputs'].shape[0]) for ex in dataset])


def pad_inputs(dataset, max_size):
    data = []

    def generator():
        for ex in data:
            yield ex

    for i, ex in enumerate(summary_train_dataset):
        for i in range(max_size - ex['inputs'].shape[0]):
            ex['inputs'] = tf.pad(ex['inputs'], paddings=[[1, 0]])
        data.append(ex)
    # stack = tf.stack(data)
    # print stack
    return tf.data.Dataset.from_generator(generator, np.int32)

# Enable TF Eager execution
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys

# Setup some directories
data_dir = os.path.expanduser("data/tensor2tensor/data")
tmp_dir = os.path.expanduser("data/tensor2tensor/tmp")
train_dir = os.path.expanduser("data/tensor2tensor/train")
checkpoint_dir = os.path.expanduser("data/tensor2tensor/checkpoints")
tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)
tf.gfile.MakeDirs(train_dir)
tf.gfile.MakeDirs(checkpoint_dir)
gs_data_dir = "gs://tensor2tensor-data"
gs_ckpt_dir = "gs://tensor2tensor-checkpoints/"

# Fetch the summary problem
summary_problem = problems.problem("summary_problem")

# extracted example is dict with keys: inputs, targets, batch_prediction_key
ex = tfe.Iterator(summary_problem.dataset(Modes.TRAIN, data_dir)).next()

# set hyper parameters to pre-set values. can easuly customize.
# transformer_prepend was suggested by the official walkthrough on:
#   https://tensorflow.github.io/tensor2tensor/walkthrough.html
hparams = trainer_lib.create_hparams(
    hparams_set="transformer_prepend",
    data_dir=data_dir,
    problem_name="summary_problem"
)

# display parameters to be used:
# for p in hparams.values():
#     print(str(p) + "  " + str(hparams.get(p)))

transformer = registry.model('transformer')(hparams, Modes.TRAIN)

# Prepare for the training loop

# In Eager mode, opt.minimize must be passed a loss function wrapped with
# implicit_value_and_gradients
@tfe.implicit_value_and_gradients
def loss_fn(features):
    _, losses = transformer(features)
    return losses["training"]

# Setup the training data
BATCH_SIZE = 128
summary_train_dataset = summary_problem.dataset(Modes.TRAIN, data_dir)
print summary_train_dataset
# what if I create my own numpy arrays each being a batch? 

import sys
sys.exit()
# max_size = largest_input_tensor_size(summary_train_dataset)
# summary_train_dataset = pad_inputs(summary_train_dataset, max_size)
# for _, ex in enumerate(summary_train_dataset):
#     print ex['inputs']

# summary_train_dataset = summary_train_dataset.repeat(None).batch(BATCH_SIZE)
summary_train_dataset = summary_train_dataset.apply(
    tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE)
)
print summary_train_dataset

optimizer = tf.train.AdamOptimizer()

# Train
NUM_STEPS = 500

# somehow I need the sensors here to be the same shape.
for count, example in enumerate(tfe.Iterator(summary_train_dataset)):
    # Make example shape 4D.
    example["targets"] = tf.reshape(
        example["targets"],
        [BATCH_SIZE, 1, 1, 1]
    )
    loss, gv = loss_fn(example)
    optimizer.apply_gradients(gv)

    if count % 50 == 0:
        print("Step: %d, Loss: %.3f" % (count, loss.numpy()))
    if count >= NUM_STEPS:
        break
