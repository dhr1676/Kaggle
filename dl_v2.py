# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, SpatialDropout1D, Dropout
from keras.layers import Embedding, GlobalMaxPool1D, BatchNormalization
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer

import logging

debug = False
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

## Set basic parameters
max_features = 1000000
maxlen = 1000
batch_size = 128
embedding_dims = 256
epochs = 6

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

## Build the model
logger.info('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(128))
model.add(BatchNormalization())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy', 'mae'])

## Print model summary
print(model.summary())

## Let's train it!
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.03) # Andrew Ng said 98:1:1 lol

## Then predict
y_pred = model.predict(x_test)

## Save the model (could be huge ~GB)
model.save('dl_feb8.h5')

## Then simply submit it
submission = pd.read_csv("./input/sample_submission.csv")
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred / 1.4
submission.to_csv("submission_bn_fasttext_feb8.csv", index=False)
