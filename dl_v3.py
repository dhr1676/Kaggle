# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, SpatialDropout1D, Dropout
from keras.layers import Embedding, GlobalMaxPool1D, BatchNormalization
from keras.layers import LSTM, Input, Bidirectional
from keras.preprocessing.text import Tokenizer

import logging
import gc
from tqdm import tqdm
import codecs

debug = False
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

## Set basic parameters
max_features = 800000
maxlen = 1500
batch_size = 32
embedding_dims = 300 ## shall not change
epochs = 6
MAX_NB_WORDS = 100000

## Load data set
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
train_df.describe()

X_train = train_df["comment_text"].fillna("sterby").values
y_train = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test_df["comment_text"].fillna("sterby").values

## Tokenize data
logger.info('Tokenizing data...')
tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(list(X_train) + list(X_test))
x_train = tok.texts_to_sequences(X_train)
x_test = tok.texts_to_sequences(X_test)

## Pad sequences
logger.info('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
logger.info('x_train shape: %s', x_train.shape)
logger.info('x_test shape: %s', x_test.shape)

## Load pre-trained model
logger.info('Loading FastText model...')
##  embedding_matrix = loadEmbeddingMatrix('fasttext', tok)

# print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('./pretrained/wiki.en.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except:
        pass
f.close()
print('found %s word vectors' % len(embeddings_index))
print('preparing embedding matrix...')
words_not_found = []
nb_words = min(MAX_NB_WORDS, len(tok.word_index))
embedding_matrix = np.zeros((nb_words, embedding_dims))
for word, i in tok.word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


## Build the model
logger.info('Build model...')

model = Sequential()
# model.add(Input(shape=maxlen, ))
model.add(Embedding(len(tok.word_index), embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False))
model.add(SpatialDropout1D(0.25))
model.add(Bidirectional(LSTM(128, return_sequences=True, name='lstm_layer', dropout=0.1, recurrent_dropout=0.1)))
model.add(BatchNormalization())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy', 'mae'])

## Print model summary

logger.info(model.summary())

## Let's train it!
logger.info('Training Model')
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                 validation_split=0.03)  # Andrew Ng said 98:1:1 lol

## Then predict
logger.info('Predicting using model')
y_pred = model.predict(x_test)

## Save the model (could be huge ~GB)
logger.info('Save model')
model.save('dl_feb14.h5')

## Then simply submit it
logger.info('Read submission then save')
submission = pd.read_csv("./input/sample_submission.csv")
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred / 1.4
submission.to_csv("submission_bn_fasttext_feb8.csv", index=False)

logger.info('Done successfully!')
