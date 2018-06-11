
# silence the future warning from h5py
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

from keras.models import Model
from keras.layers import Embedding, Dense, Input, Concatenate, add, RepeatVector, concatenate
from keras.layers.recurrent import LSTM
from keras import backend as K
K.set_image_dim_ordering('tf')
"""
Compilation functions for models to be tested.
Takes in various configurations required by all models and returns a compiled
    Keras model.
"""


def compile_model_0(sum_model, embedding_matrix, vocab_size, max_sen_len):
    inputs1 = Input(shape=(sum_model.max_input_seq_length,))
    am1 = Embedding(
        input_dim=vocab_size,
        output_dim=sum_model.glove_dim,
        weights=[embedding_matrix],
        input_length=max_sen_len,
        trainable=False
    )(inputs1)
    am2 = LSTM(sum_model.glove_dim)(am1)

    inputs2 = Input(shape=(sum_model.max_target_seq_length,))
    sm1 = Embedding(
        input_dim=vocab_size,
        output_dim=sum_model.glove_dim,
        weights=[embedding_matrix],
        input_length=max_sen_len,
        trainable=False
    )(inputs2)
    sm2 = LSTM(sum_model.glove_dim)(sm1)

    decoder1 = Concatenate()([am2, sm2])
    outputs = Dense(sum_model.num_target_tokens, activation='softmax')(decoder1)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def compile_model_1(sum_model, embedding_matrix, vocab_size, max_sen_len):
    # article input model
    inputs1 = Input(shape=(sum_model.max_input_seq_length,))
    article1 = Embedding(
        input_dim=vocab_size,
        output_dim=sum_model.glove_dim,
        weights=[embedding_matrix],
        input_length=max_sen_len,
        trainable=False
    )(inputs1)
    article2 = LSTM(sum_model.glove_dim)(article1)
    article3 = RepeatVector(sum_model.max_target_seq_length)(article2)

    # summary input model
    inputs2 = Input(shape=(sum_model.max_target_seq_length,))
    summ1 = Embedding(
        input_dim=vocab_size,
        output_dim=sum_model.glove_dim,
        weights=[embedding_matrix],
        input_length=sum_model.max_target_seq_length,
        trainable=False
    )(inputs2)

    # decoder model
    decoder1 = concatenate([article3, summ1])
    decoder2 = LSTM(sum_model.glove_dim)(decoder1)
    outputs = Dense(sum_model.num_target_tokens, activation='softmax')(decoder2)

    # tie it together [article, summary] -> [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
