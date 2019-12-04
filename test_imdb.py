import logging
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dictionary.dictionary import Dictionary
from text.text_util import Text_Util

from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


def test_model_kfold(model_name, model, X, y, CV=5):
    k_fold = StratifiedKFold(n_splits=CV, shuffle=True)
    out_of_fold_predictions = np.zeros((X.shape[0],))
    scores = []
    for train_index, test_index in k_fold.split(X, y):
        model.fit(X[train_index], y[train_index])
        y_pred = model.predict(X[test_index])
        scores.append(np.mean(y_pred == y[test_index]))
        out_of_fold_predictions[test_index] = y_pred
    scores=np.array(scores)
    return scores.mean(), scores.std()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TFIDF = False
TOKENIZED = 0

dslcc_train_path = './data/dslcc/DSL-TRAIN.txt'
dslcc_test_path = './data/dslcc/DSL-DEV.txt'
imdb_train_path = './data/imdb/train_data_set.csv'
imdb_test_path = './data/imdb/train_data_set.csv'

    
imdb_train = pd.read_csv(imdb_train_path, names=['review', 'sentiment'])
imdb_test = pd.read_csv(imdb_train_path, names=['review', 'sentiment'])
X = imdb_train.review.to_numpy()
y = imdb_train.sentiment.to_numpy()
X_test = imdb_test.review.to_numpy()
y_test = imdb_test.sentiment.to_numpy()
#print('raw y is ', y)
#print('raw values are ', imdb_train.sentiment.values)
logger.info('### Creating dictionary...')
text_util = Text_Util()
sorted_labels = np.unique(y)
#print('Sorted labels are ', sorted_labels)
dic = Dictionary(sorted_labels)

# Preprocessing text
logger.info('### Preprocessing text...')
text_util = Text_Util()
X = text_util.get_preprocessed_tokenized_sentences(X)

# Updating dictionary
logger.info('### Updating dictionary...')
for i, elem in enumerate(X):
    dic.update_tokenized(elem, y[i])
selected_words = {}
for i, l in enumerate(sorted_labels):
  u_list = dic.get_n_words_unique_to_label(l, 1000)
  o_list = dic.get_n_top_words_given_label(l, 5000)
  for u in u_list:
      selected_words[u] = True
  for o in o_list:
      selected_words[o] = True
  print(f"Size of selected words set {len(selected_words)}")

x = []
for tokenized_comment in X:
    x.append(np.array([w for w in tokenized_comment if w in selected_words]))
#X = np.array(x)
x = list(map(' '.join, x))
#print('Now X is ....', X)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(x)


X_test = text_util.get_preprocessed_tokenized_sentences(X_test)
x = []
for tokenized_comment in X_test:
    x.append(np.array([w for w in tokenized_comment if w in selected_words]))
x = list(map(' '.join, x))
X_test = vectorizer.transform(x)

print('Splitting the data...')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, shuffle=True, stratify=y)

logger.info(f'### X.shape {X.shape}')
logger.info(f'### y.shape {y.shape}')
logger.info(f'### X_test.shape {X_test.shape}')
logger.info(f'### y_test.shape {y_test.shape}')
classifiers  =  {}
classifiers['SVC'] = SVC(gamma='auto')
classifiers['NuSVC'] = NuSVC(gamma='auto')
classifiers['LinearSVC'] = LinearSVC()
classifiers['SGDClassifier'] = SGDClassifier()

print('Calculating validation scores...')
for k, v in classifiers.items():
    print('### Testing with model ', k)
    print('Fitting ...')
    v.fit(X_train, y_train)
    print('Predicting ...')
    y_pred = v.predict(X_val)
    print('Accuracy: %f' %(np.mean(y_pred == y_val)))
    for l in sorted_labels:
        indexes = np.argwhere(y_val == l)
        print('Accuracy for label %s: %f' %(l, np.mean(y_pred[indexes] == y_val[indexes])))
    print(classification_report(y_pred, y_val))

print('Calculating test scores...')
for k, v in classifiers.items():
    print('### Testing with model ', k)
    print('Fitting ...')
    v.fit(X, y)
    print('Predicting ...')
    y_pred = v.predict(X_test)
    print('Accuracy: %f' %(np.mean(y_pred == y_test)))
    for l in sorted_labels:
        indexes = np.argwhere(y_test == l)
        print('Accuracy for label %s: %f' %(l, np.mean(y_pred[indexes] == y_test[indexes])))
    print(classification_report(y_pred, y_test))
    
