# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:49:46 2019

@author: Yassir
"""

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from classifier import Classifier_Util
from sklearn.ensemble import AdaBoostClassifier



import matplotlib.pyplot as plt 
import numpy as np

# Loading data
train_set_path = './data/data_train.pkl'
test_set_path = './data/data_test.pkl'
save_model_path = './model.pkl'


# Experiment with countVectorizer + TfidfTransformer
print('### Begin experimentation')
class_tool =  Classifier_Util()
print('Loading model... ')
#class_tool.load_model(save_model_path)

# setting models to process features

class_tool.set_word_vectorizer(TfidfVectorizer())
class_tool.set_feature_selector(SelectKBest(chi2, 20000))

print('Preprocessing raw data...')
# Preprocessing text
X, y = class_tool.preprocess_text(train_set_path)


# Instantiating models
models = {}
hypers = {}


# Best models
models['Naive_Bayes'] = MultinomialNB(alpha = 0.3)
models['Comp_Naive_Bayes'] = ComplementNB(alpha = 1.0)
models['SVM'] = SGDClassifier(penalty='l2', loss='hinge', alpha=0.001)
models['KNN'] = KNeighborsClassifier(n_neighbors=150)
models['logistic_regression'] = LogisticRegression(penalty='l2', dual=False)
models['Random_forest_1000'] = RandomForestClassifier(n_estimators = 1000, max_features='auto')


#models['mlp_250'] = MLPClassifier(hidden_layer_sizes=(250, ), activation='relu', solver='adam', alpha=0.0001, batch_size=200,
#                                learning_rate='adaptive', shuffle=True, early_stopping=True, validation_fraction=.1,
#                                verbose=True, max_iter=2)


#models['mlp_250'] = MLPClassifier(hidden_layer_sizes=(250,), solver='adam', activation='tanh', alpha=0.0001)

#models['Gradient Boosting Classifier'] = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features="auto", max_depth = 2, random_state = 0)
#models['Gradient Bagging Classifier'] = BaggingClassifier(n_estimators=20)
#models['Random Forest'] = RandomForestClassifier(n_estimators=1000, max_depth=5)
#models['Linear Support Vector Machine with SGD l2 hinge'] = SGDClassifier(penalty='l2', loss='hinge', max_iter=5000)
#models['Linear Support Vector Machine l1'] = LinearSVC(penalty='l1', loss='squared_hinge', dual = False, max_iter=5000)
#models['Linear Support Vector Machine l2 sq-hinge'] = LinearSVC(penalty='l2', loss='squared_hinge', max_iter=5000)
#models['Linear Support Vector Machine l2 hinge'] = LinearSVC(penalty='l2', loss='hinge', max_iter=100000)
#models['Linear Support Vector Machine with SGD l1'] = SGDClassifier(penalty='l1', loss='squared_hinge', max_iter=5000)
#models['Linear Support Vector Machine with SGD l2 sq-hinge'] = SGDClassifier(penalty='l2', loss='squared_hinge', max_iter=5000)
#models['Linear Support Vector Machine with SGD l2 hinge'] = SGDClassifier(penalty='l2', loss='hinge', max_iter=5000)
#models['Support Vector Machine'] = SVC(kernel='poly') #Infinite loop: can't work with non normalized data
#models['Random Forest'] = RandomForestClassifier(n_estimators=200, max_depth=5)

# Testing
'''
print('Testing 10 times...')
scores = np.zeros((20, 2))
for i in range(20):
'''
# Splitting data
print('Splitting data...')
X_train, X_test, y_train, y_test = class_tool.split_data(X, y, 0.2)

# Creating words vectors
print('Creating word vectors...')
class_tool.fit_words_vectorizer(X_train, y_train)
X_train = class_tool.get_words_vector(X_train)
X_test = class_tool.get_words_vector(X_test)

# Feature selection
#class_tool.fit_feature_selector(X_train, y_train)
#print('Selecting features...')
#X_train = class_tool.get_selected_features(X_train)
#X_test = class_tool.get_selected_features(X_test)


'''
accs_train = []
accs_test = []

interval = (np.linspace(1, 200, 10)).astype(int)
for k in interval:
    model = KNeighborsClassifier(n_neighbors=k)
    acc_train, acc_test = class_tool.fit_model('KNN', model, X_train, X_test, y_train, y_test)
    accs_train.append(acc_train)
    accs_test.append(acc_test)


plt.plot(interval, accs_train)
plt.xlabel('Values of K')
plt.ylabel('Train accuracy')
plt.plot(interval, accs_test)
plt.ylabel('Test accuracy')
plt.show()
plt.savefig('knn_train_test_accuracy.png') 
'''
clf1 = MultinomialNB(alpha = 0.3)
clf2 = ComplementNB(1.0)
clf3 = SGDClassifier(penalty='l2', loss='hinge', max_iter=2000, alpha=0.0001)
clf4 = MLPClassifier(hidden_layer_sizes=(250,), solver='adam', activation='relu', alpha=0.0001, max_iter=2)
clf5 = KNeighborsClassifier(n_neighbors=160)
clf6 = RandomForestClassifier(n_estimators = 1000, max_depth = 5, max_features='auto')
'''
voting_models = {}
voting_models['MNB-SVM'] = VotingClassifier(estimators=[('MNB', clf1), ('SVM', clf3)], voting='hard')
voting_models['MNB-MLP'] = VotingClassifier(estimators=[('MNB', clf1), ('MLP', clf4)], voting='hard')
voting_models['MNB-KNN'] = VotingClassifier(estimators=[('MNB', clf1), ('KNN', clf5)], voting='hard')
voting_models['MNB-RF'] = VotingClassifier(estimators=[('MNB', clf1), ('RF', clf6)], voting='hard')
voting_models['SVM-MLP'] = VotingClassifier(estimators=[('SVM', clf3), ('MLP', clf4)], voting='hard')
voting_models['SVM-KNN'] = VotingClassifier(estimators=[('SVM', clf3), ('KNN', clf5)], voting='hard')
voting_models['SVM-RF'] = VotingClassifier(estimators=[('SVM', clf3), ('RF', clf6)], voting='hard')
voting_models['MLP-KNN'] = VotingClassifier(estimators=[('MLP', clf4), ('KNN', clf5)], voting='hard')
voting_models['MLP-RF'] = VotingClassifier(estimators=[('MLP', clf4), ('RF', clf6)], voting='hard')
voting_models['MNB-SVM-MLP'] = VotingClassifier(estimators=[('MNB', clf1), ('SVM', clf3), ('MLP', clf4)], voting='hard')
voting_models['MNB-SVM-KNN'] = VotingClassifier(estimators=[('MNB', clf1), ('SVM', clf3), ('KNN', clf5)], voting='hard')
voting_models['MNB-SVM-RF'] = VotingClassifier(estimators=[('MNB', clf1), ('SVM', clf3), ('RF', clf6)], voting='hard')
voting_models['MNB-MLP-KNN'] = VotingClassifier(estimators=[('MNB', clf1), ('MLP', clf4), ('KNN', clf5)], voting='hard')
voting_models['MNB-MLP-RF'] = VotingClassifier(estimators=[('MNB', clf1), ('MLP', clf4), ('RF', clf6)], voting='hard')
voting_models['MNB-KNN-RF'] = VotingClassifier(estimators=[('MNB', clf1), ('KNN', clf5), ('RF', clf6)], voting='hard')
voting_models['MNB-SVM-MLP-KNN'] = VotingClassifier(estimators=[('MNB', clf1), ('SVM', clf3), ('MLP', clf4), ('KNN', clf5)], voting='hard')
voting_models['MNB-SVM-MLP-RF'] = VotingClassifier(estimators=[('MNB', clf1), ('SVM', clf3), ('KNN', clf5), ('RF', clf6)], voting='hard')
voting_models['MNB-SVM-MLP-KNN-RF'] = VotingClassifier(estimators=[('MNB', clf1), ('SVM', clf3), ('MLP', clf4), ('KNN', clf5), ('RF', clf6)], voting='hard')
voting_models['MNB-SVM-MLP-CNB'] = VotingClassifier(estimators=[('MNB', clf1), ('SVM', clf3), ('MLP', clf4), ('CNB', clf2)], voting='hard')
for key,val in voting_models.items():
    class_tool.set_voting_classifier(val)
    class_tool.test_voting_models(key, X_train, X_test, y_train, y_test)

'''
for key,val in models.items():
    class_tool.test_model(key, val, X_train, X_test, y_train, y_test)

print('### End of experiment')


