import os
import sys
import logging
import random
import pickle

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import metrics
import numpy as np

from feature_extraction.dataset_reader import DatasetProcessor
from utils import get_kfold_with_seacrh_criteria

DATA_PATH = 'data'
RANDOM_SEED = 42
DUMP_PATH = os.path.join(DATA_PATH, 'data.pkl')
USE_DUMPED_DATA = True
USE_SCALER = False
DOWNSAMPLE = False


def weighted_accuracy(y_true, y_pred):
    weights = [2 if y == 2 else 1 for y in y_true]
    return np.sum(weights * (y_true == y_pred)) / sum(weights)


def set_seed(random_state=RANDOM_SEED):
    random.seed(random_state)
    np.random.seed(random_state)


if __name__ == '__main__':
    # if you face troubles with importing spacy vocab: exec "python -m spacy download en"

    set_seed()

    logging.basicConfig(filename="logs.log", filemode='w', level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('Processing data')
    data = DatasetProcessor(train_path=os.path.join(DATA_PATH, 'train_df.pkl'),
                            test_path=os.path.join(DATA_PATH, 'test_df.pkl'),
                            validate_pipe=False)

    if USE_DUMPED_DATA and os.path.exists(DUMP_PATH):
        logging.info('Will use dumped data')
        with open(DUMP_PATH, 'rb') as f:
            data = pickle.load(f)
    else:
        data.get_vectorization()

        with open(DUMP_PATH, 'wb') as f:
            pickle.dump(data, f)

    train = data.train
    X = data.train_x
    y = data.train_y
    X_test = data.test_x

    with open('data/train_bert_2.pickle', 'rb') as handle:
        train_bert = pickle.load(handle)

    with open('data/test_bert_2.pickle', 'rb') as handle:
        test_bert = pickle.load(handle)

    X_feat = train_bert['train_x']
    X_test_feat = test_bert['test_x']

    feat_to_take = [col for col in list(X_feat) if not col.startswith(('id', 'ap', 'rp'))]
    X_feat = X_feat[feat_to_take]
    X_test_feat = X_test_feat[feat_to_take]

    X_feat = np.array(X_feat)
    X_test_feat = np.array(X_test_feat)

    X = np.concatenate((X.toarray(), X_feat), axis=1)
    X_test = np.concatenate((X_test.toarray(), X_test_feat), axis=1)

    if DOWNSAMPLE:
        logging.info('Performing dataset downsampling.')
        idx_02 = [idx for idx in range(len(y)) if y[idx] in (0, 2)]
        idx_1 = random.sample([idx for idx in range(len(y)) if y[idx] == 1], len(idx_02))
        idx = idx_02 + idx_1
        X = X[idx, :]
        y = np.array(y[idx])

    logging.info('Splitting to CV splits')
    kf = KFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
    kf.get_n_splits(X)
    folds = get_kfold_with_seacrh_criteria(train)

    mm_scaler = preprocessing.MaxAbsScaler()

    waccs = []
    fold = 1
    for train_index, test_index in kf.split(X):
    # for train_index, test_index in folds:
        logging.info('FOLD #{}'.format(fold))
        fold += 1
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        if USE_SCALER:
            X_train = mm_scaler.fit_transform(X_train)
            X_val = mm_scaler.transform(X_val)

        logging.info('START OF MODEL FIT')
        # classifier = RandomForestClassifier(min_samples_split=2, n_jobs=4)
        classifier = LogisticRegression(max_iter=1000)
        # classifier = SVC()
        classifier.fit(X_train, y_train)
        logging.info('END OF MODEL FIT')

        pred_val = classifier.predict(X_val)
        w_acc = weighted_accuracy(y_val, pred_val)
        logging.info('Validation accuracy: {}'.format(metrics.accuracy_score(y_val, pred_val)))
        logging.info('Validation waccuracy: {}'.format(w_acc))
        waccs.append(w_acc)

    logging.info('Fitting final model on whole dataset')
    # classifier = RandomForestClassifier(min_samples_split=2, n_jobs=4)
    classifier = LogisticRegression(max_iter=1000)
    # classifier = SVC()
    if USE_SCALER:
        X = mm_scaler.fit_transform(X)
        X_test = mm_scaler.transform(X_test)

    classifier.fit(X, y)
    mean_wacc = np.mean(waccs)
    logging.info('MEAN OF weighted accuracies: {}'.format(mean_wacc))

    logging.info('Predicting test set values')
    pred_test = classifier.predict(X_test)
    data.get_submission_csv(pred_test, mean_wacc)
