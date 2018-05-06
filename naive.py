import numpy as np
import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn import svm
from sklearn.neural_network import MLPRegressor

from joblib import Parallel, delayed
import multiprocessing
from functools import partial

import logging


def pre_process_input(t):
    t = t.strip()
    z = re.findall(r'[A-Za-z]+', t)
    z = [a for a in z if len(a) > 2]
    word_net_lemmatizer = nltk.stem.WordNetLemmatizer()
    z = [word_net_lemmatizer.lemmatize(a) for a in z]
    z = [a for a in z if not a in stop_words]
    t = ' '.join(z)
    return t


def log_reg(train, test, metrics_df, cols):
    # Initialize regression
    clf = LogisticRegression(tol=1e-4, solver='saga')
    # Train the logistic regression
    logger.debug('Training logistic regression for %s', cols)
    clf.fit(train, metrics_df[cols])
    # Predict the test set and train set (to testify)
    logger.debug('Predicting by logistic regression...')
    predicted_train = clf.predict_proba(train_vectorized)[:, 1]
    predicted_test = clf.predict_proba(test)[:, 1]
    logger.info('log loss: %.5f from column %s', log_loss(metrics_df[cols], predicted_train), cols.upper())
    return predicted_test


def svm_reg(train, test, metrics_df, cols):
    # Initialize regression
    svr_rbf = svm.SVR()
    # Train the SVM
    logger.debug('Training SVM regression for %s...', cols)
    svr_rbf.fit(train, metrics_df[cols])
    # Predict the test set and train set (to testify)
    logger.debug('Predicting by SVM regression...')
    predicted_test = svr_rbf.predict(test)
    predicted_train = svr_rbf.predict(train)
    logger.info('log loss: %.5f from column %s', log_loss(metrics_df[cols], predicted_train), cols.upper())
    return predicted_test


def nn_class(train, test, metrics_df, cols):
    clf = MLPRegressor()
    logger.debug('Training NN regression for %s...', cols)
    clf.fit(train, metrics_df[cols])
    logger.debug('Predicting by NN regression...')
    predicted_test = clf.predict(test)
    predicted_train = clf.predict(train)
    logger.info('log loss: %.5f from column %s', log_loss(metrics_df[cols], predicted_train), cols.upper())
    return predicted_test

if __name__ == "__main__":

    # Basic configurations
    debug = False
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    verbose_level = 10

    if debug:
        num_cores = 1
    else:
        num_cores = multiprocessing.cpu_count()

    # Load dataset and stop words (e.g. am is are we he she etc)
    logger.info('Start reading training set and stop words...')
    if debug:
        train_set = pd.read_csv('./input/train_trimmed.csv')
    else:
        train_set = pd.read_csv('./input/train.csv')
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Get rid of NaN, Null etc
    null_text = train_set.comment_text[2]
    pre_process_input(null_text)

    # Grab matrics
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    metrics = train_set[columns]
    train_set.drop(['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)

    # Use NLTK to rip off unrelated words
    logger.info('Filter out stop words of training set using lemmatizer...')
    train_set.comment_text = Parallel(n_jobs=num_cores, verbose=verbose_level)(
        delayed(pre_process_input)(t)for t in train_set.comment_text)

    # Initialize vectorizer and apply it to training set
    vectorizer = TfidfVectorizer(min_df=3, max_df=0.8,
                           ngram_range=(1, 2),
                           strip_accents='unicode',
                           smooth_idf=True,
                           sublinear_tf=True)
    logger.info('Applying vectorizer to training set...')
    vectorizer = vectorizer.fit(train_set['comment_text'])
    train_vectorized = vectorizer.transform(train_set['comment_text'])

    # Do the same to test set
    logger.info('Start reading test set...')
    if debug:
        test_set = pd.read_csv('./input/test_trimmed.csv')
    else:
        test_set = pd.read_csv('./input/test.csv')
    test_set.fillna(value=null_text, inplace=True)
    test_set.drop(['id'], axis=1, inplace=True)

    logger.info('Filter out stop words of test set using lemmatizer...')
    test_set.comment_text = Parallel(n_jobs=num_cores, verbose=verbose_level)(
        delayed(pre_process_input)(t) for t in test_set.comment_text)

    logger.info('Applying vectorizer to test set...')
    test_vectorized = vectorizer.transform(test_set['comment_text'])

    # Prepare submission
    if debug:
        submission = pd.read_csv('./input/sample_submission_trimmed.csv')
    else:
        submission = pd.read_csv('./input/sample_submission.csv')

    # Train regression and predict then write to file
    # Parallel for submission
    if debug:
        for col in metrics:
            submission[col] = log_reg(train_vectorized, test_vectorized, metrics, col)
    else:
        test_set_predicted = Parallel(n_jobs=num_cores, verbose=verbose_level)(delayed(log_reg)(
            train_vectorized, test_vectorized, metrics, col) for col in metrics)
        for i in range(0, 6):
            submission[columns[i]] = test_set_predicted[i]

    submission.to_csv("my_submission_trimmed.csv", index=False)
