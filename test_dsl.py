import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import cross_val_score
from dictionary.dictionary import Dictionary
from text.text_util import Text_Util




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
'''
le_fr = LabelEncoder()
le_sp = LabelEncoder()
le_fr_sp = LabelEncoder()
'''

KEYS = ['FR', 'SP', 'FR-SP']

### TRAIN
train_dsl = pd.read_csv('./data/dslcc/DSL-TRAIN.txt', sep='\t', names=['comment', 'lang'])
### TEST
test_dsl = pd.read_csv('./data/dslcc/DSL-DEV.txt', sep='\t', names=['comment', 'lang'])

# Dictionaries
train_dic = {}
test_dic = {}


# FRENCH
fr_train_dsl = train_dsl.query('lang == "fr-FR" or lang == "fr-CA"', inplace=False)
fr_test_dsl = test_dsl.query('lang == "fr-FR" or lang == "fr-CA"', inplace=False)
train_dic['FR'] = (fr_train_dsl.comment.to_numpy(), fr_train_dsl.lang.to_numpy())
test_dic['FR'] = (fr_test_dsl.comment.to_numpy(), fr_test_dsl.lang.to_numpy())
 
# SPANISH
sp_train_dsl = train_dsl.query('lang == "es-AR" or lang == "es-ES" or lang == "es-PE"', inplace=False)
sp_test_dsl = test_dsl.query('lang == "es-AR" or lang == "es-ES" or lang == "es-PE"', inplace=False)
train_dic['SP'] = (sp_train_dsl.comment.to_numpy(), sp_train_dsl.lang.to_numpy())
test_dic['SP'] = (sp_test_dsl.comment.to_numpy(), sp_test_dsl.lang.to_numpy())

# FRENCH - SPANISH
fr_sp_train_dsl = train_dsl.query('lang == "es-AR" or lang == "es-ES" or lang == "es-PE" or lang == "fr-FR" or lang == "fr-CA"', inplace=False)
fr_sp_test_dsl = test_dsl.query('lang == "es-AR" or lang == "es-ES" or lang == "es-PE" or lang == "fr-FR" or lang == "fr-CA"', inplace=False)
train_dic['FR-SP'] = (fr_sp_train_dsl.comment.to_numpy(), fr_sp_train_dsl.lang.to_numpy())
test_dic['FR-SP'] = (fr_sp_test_dsl.comment.to_numpy(), fr_sp_test_dsl.lang.to_numpy())


# Models to test
classifiers  =  {}
classifiers['SVC'] = SVC(gamma='auto')
classifiers['NuSVC'] = NuSVC(gamma='auto')
classifiers['LinearSVC'] = LinearSVC()
classifiers['SGDClassifier'] = SGDClassifier()
text_util = Text_Util()
##### MAIN LOOP
print('Main Loop ...')
for k in KEYS:
    print('################ Testing %s language(s) ################' %k)
    
    # Preprocessing text
    logger.info('### Preprocessing text...')
   
    # Train data
    (X, y) = train_dic[k]
    # Test data
    (X_test, y_test) = test_dic[k]
    
    logger.info('### Creating dictionary...')
    sorted_labels = np.unique(y)
    dic = Dictionary(sorted_labels)
    X = text_util.get_preprocessed_tokenized_sentences_dsl(X)
    
    # Updating dictionary
    logger.info('### Updating dictionary...')
    for i,x in enumerate(X):
        dic.update_tokenized(x, y[i])
    
    selected_words = {}
    for i, l in enumerate(sorted_labels):
        u_list = dic.get_n_words_unique_to_label(l, 1000)
        o_list = dic.get_n_top_words_given_label(l, 5000)
        for u in u_list:
            selected_words[u] = True
        for o in o_list:
            selected_words[o] = True
    
    #print(f"Size of selected words set {len(selected_words)}")
    x = []
    for tokenized_comment in X:
        x.append(np.array([w for w in tokenized_comment if w in selected_words]))
    x = list(map(" ".join, x))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(x)
    
    X_test = text_util.get_preprocessed_tokenized_sentences_dsl(X_test) 
    x = []
    for tokenized_comment in X_test:
        x.append(np.array([w for w in tokenized_comment if w in selected_words]))
    x = list(map(" ".join, x))
    X_test = vectorizer.transform(x)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, shuffle=True, stratify=y)
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
            #print('found %d entries for language %s' %(len(indexes), l))
            print('Accuracy for language %s: %f' %(l, np.mean(y_pred[indexes] == y_val[indexes])))
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
            #print('found %d entries for language %s' %(len(indexes), l))
            print('Accuracy for language %s: %f' %(l, np.mean(y_pred[indexes] == y_test[indexes])))
        print(classification_report(y_pred, y_test))
