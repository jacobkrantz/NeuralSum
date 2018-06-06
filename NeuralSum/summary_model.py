
from abstract_model import AbstractModel
from config import config

# silence the future warning from h5py
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import os
import numpy as np
import logging as log

# ignore tensorflow console output
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Embedding, Dense, Input, Concatenate, add
from keras.layers.recurrent import LSTM
from keras import backend as K
K.set_image_dim_ordering('tf')

class SummaryModel(AbstractModel):
    def __init__(self, m_params):
        log.getLogger()
        self.num_input_tokens = m_params['num_input_tokens']
        self.max_input_seq_length = m_params['max_input_seq_length']
        self.num_target_tokens = m_params['num_target_tokens']
        self.max_target_seq_length = m_params['max_target_seq_length']
        self.input_word2idx = m_params['input_word2idx']
        self.input_idx2word = m_params['input_idx2word']
        self.target_word2idx = m_params['target_word2idx']
        self.target_idx2word = m_params['target_idx2word']
        self.m_params = m_params
        self.model = None

    def compile(self, embeddings, vocab, max_sen_len, max_sum_len):
        embedding_matrix = self._make_embedding_matrix(
            embeddings,
            vocab
        )
        inputs1 = Input(shape=(self.max_input_seq_length,))
        am1 = Embedding(
            len(vocab),
            config['glove_dim'],
            weights=[embedding_matrix],
            input_length=max_sen_len,
            trainable=False)(inputs1)
        am2 = LSTM(config['glove_dim'])(am1)

        inputs2 = Input(shape=(self.max_target_seq_length,))
        sm1 = Embedding(
            len(vocab),
            config['glove_dim'],
            weights=[embedding_matrix],
            input_length=max_sen_len,
            trainable=False)(inputs2)
        sm2 = LSTM(config['glove_dim'])(sm1)

        decoder1 = Concatenate()([am2, sm2])
        outputs = Dense(self.num_target_tokens, activation='softmax')(decoder1)

        self.model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, Xtrain, Xtest, Ytrain, Ytest):
        config_filename = config['model_output_path'] + config['model_config_file']
        weight_file_path = config['model_output_path'] + config['model_weight_file']
        architecture_file_path = config['model_output_path'] + config['model_architecture_file']

        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_filename, self.m_params)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self._split_target_text(Ytrain)
        Ytest = self._split_target_text(Ytest)

        Xtrain = self._transform_input_text(Xtrain)
        Xtest = self._transform_input_text(Xtest)

        train_gen = self._generate_batch(Xtrain, Ytrain, config['batch_size'])
        test_gen = self._generate_batch(Xtest, Ytest, config['batch_size'])

        total_training_samples = sum([len(target_text) - 1 for target_text in Ytrain])
        total_testing_samples = sum([len(target_text) - 1 for target_text in Ytest])
        train_num_batches = total_training_samples // config['batch_size']
        test_num_batches = total_testing_samples // config['batch_size']

        history = self.model.fit_generator(
            generator=train_gen,
            steps_per_epoch=train_num_batches,
            epochs=config['epochs'],
            verbose=config['verbose'],
            validation_data=test_gen,
            validation_steps=test_num_batches,
            callbacks=[checkpoint]
        )
        self.model.save_weights(weight_file_path)
        return history

    def test(self, articles, num_to_test=-1):
        if num_to_test == -1:
            for art in articles:
                art.generated_summary = self.test_single(art.sentence)
            return articles

        for i in range(num_to_test):
            print('')
            print 'Sentence:'
            print articles[i].sentence
            print 'Generated Summary:'
            print self.test_single(articles[i].sentence)
            print 'Gold Standard Summaries:'
            for j, gold_sum in enumerate(articles[i].gold_summaries):
                print(str(j) + ": " + gold_sum)
            print('')

    def test_single(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)

        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_input_seq_length)
        start_token = self.target_word2idx['START']
        wid_list = [start_token]
        sum_input_seq = pad_sequences([wid_list], self.max_target_seq_length)
        terminated = False

        target_text = ''
        while not terminated:
            output_tokens = self.model.predict([input_seq, sum_input_seq])
            sample_token_idx = np.argmax(output_tokens[0, :])
            sample_word = self.target_idx2word[sample_token_idx]
            wid_list = wid_list + [sample_token_idx]

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or len(wid_list) >= self.max_target_seq_length:
                terminated = True
            else:
                sum_input_seq = pad_sequences([wid_list], self.max_target_seq_length)
        return target_text.strip()

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)
        else:
            log.critical("Incorrect weight file path: " + weight_file_path)

    def _make_embedding_matrix(self, embeddings, vocab):
        # Create a weight matrix for words in the vocab
        oov_words = set()
        embedding_matrix = np.zeros((len(vocab), config["glove_dim"]))
        for i, word in enumerate(vocab):
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                oov_words.add(word)

        if(len(oov_words) > 0):
            oov_num = str(len(oov_words))
            log.warn(oov_num + " training words not represented in GloVe embeddings.")

        return embedding_matrix

    def _split_target_text(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) + 1 >= self.max_target_seq_length:
                    x.append('END')
                    break
            temp.append(x)
        return temp

    def _transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.input_word2idx:
                    wid = self.input_word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def _generate_batch(self, x_samples, y_samples, batch_size):
        encoder_input_data_batch = []
        decoder_input_data_batch = []
        decoder_target_data_batch = []
        line_idx = 0
        while True:
            for recordIdx in range(0, len(x_samples)):
                target_words = y_samples[recordIdx]
                x = x_samples[recordIdx]
                decoder_input_line = []

                for idx in range(0, len(target_words) - 1):
                    w2idx = 0  # default [UNK]
                    w = target_words[idx]
                    if w in self.target_word2idx:
                        w2idx = self.target_word2idx[w]
                    decoder_input_line = decoder_input_line + [w2idx]
                    decoder_target_label = np.zeros(self.num_target_tokens)
                    w2idx_next = 0
                    if target_words[idx + 1] in self.target_word2idx:
                        w2idx_next = self.target_word2idx[target_words[idx + 1]]
                    if w2idx_next != 0:
                        decoder_target_label[w2idx_next] = 1
                    decoder_input_data_batch.append(decoder_input_line)
                    encoder_input_data_batch.append(x)
                    decoder_target_data_batch.append(decoder_target_label)

                    line_idx += 1
                    if line_idx >= batch_size:
                        yield [
                            pad_sequences(encoder_input_data_batch, self.max_input_seq_length),
                            pad_sequences(decoder_input_data_batch, self.max_target_seq_length)
                        ], np.array(decoder_target_data_batch)
                        line_idx = 0
                        encoder_input_data_batch = []
                        decoder_input_data_batch = []
                        decoder_target_data_batch = []
