# Sentiment analysis and similar language discrimination

## Introduction
In this project, we present the results of several experiments that consisted in using a set of classifiers on different data sets to compare their performance and characteristics. We leverage this perspective to provide some insights about some interesting questions, which were raised during our experiments.
These relate to the appropriateness of different classifiers to solve two different classification problems, the usefulness of training on all the available data even when we only need to classify a subset of labels, and finally, the effectiveness of text pre-processing of text data in different languages. At last but not least, we are also interested in exploring text data in languages that differ from English.

## Datasets
As first part of the project, we have identified two data sets:
* **Large Movie Review Data Set:** this is a data set for binary sentiment classification containing 25000 highly polar movie reviews for training, and 25000 for testing. Additional information can be found [here](http://ai.stanford.edu/~amaas/data/sentiment/).
* **The Domain Specific Languages Corpus Collection (DSLCC):** this is a multilingual collection of short excerpts of journalistic texts. It has been used as the main data set for the DSL shared tasks organized within the scope of the workshop on NLP for Similar languages, Varieties and Dialects (VarDial). Additional information on the data set is available [here](https://sites.google.com/view/vardial2019).

For the former dataset, we perform classification on the reviews (positive or negative). For the latter, our task is to distinguish between different variants of the same languages. We focus on Spanish (Argentinian, Peninsular and Peruvian) and French (Canadian and Mainland France). This consists of 75000 training samples and 9700 test samples. It should also be noted that the labels are balanced in both data sets.

## Approach
We train and measure performance of the following models: Logistic regression, Linear SVM, NuSVM, Naive Bayes and MLP. We have chosen these models because they are widely used for classification tasks. We also are interested in comparing how different feature selection techniques affect models performances in the context of two different problems.

## Feature design
We apply different pre-processing steps to each data set:
| **IMDB** | **DSLCC** |
| :------- |:--------- |
| Convert to lowercase | Convert to lowercase |
| Remove non-alphanumerical symbols | Remove non-alphanumerical symbols |
| Tokenize the comments | Tokenize the comments|
| Remove the stop words | Remove the French stop words|
| Lemmatization |  |

## Model training & hyper-parameter tuning
As we have stated before, our main focus for this project was on five classifiers: Logistic regression, Linear SVM, NuSVM, Naive Bayes and MLP. We have used Scikit Learn and Keras implementations of those models for our experiments. Regarding the data, we used the same pre-processing steps for all the models, and we split the data into train and validation sets (80%, 20%) for cross-validation for hyper-parameter tuning. 

## Experiments and results
We tuned hyper-parameters for every model via cross-validation, and all of our results are calculated against the separate test data that was provided. 

### Overall accuracy comparison
<p float="left">
<img src="https://user-images.githubusercontent.com/20977650/222184085-f887a6bc-b3bd-4fcb-82b6-75a41a56c165.png" width=47% height=47%  hspace=10>
<img src="https://user-images.githubusercontent.com/20977650/222185568-c5408228-aa6a-4d4b-bf98-56fce10a8944.png" width=47% height=47%  hspace=10>
</p>

### DSLCC classification experiments
<p float="left">
<img src="https://user-images.githubusercontent.com/20977650/222197502-352f9071-1916-4ff1-b702-3349b9085f09.png" width=47% height=41%  hspace=5>
<img src="https://user-images.githubusercontent.com/20977650/222197526-7ef16505-e87b-46f2-941d-6b89ee73542d.png" width=47% height=41%  hspace=5>
</p>

### DSLCC pre-processing experiment
<p align="center">
<img src="https://user-images.githubusercontent.com/20977650/222241049-8dd00d5e-d7a5-4928-89cb-f8fea50d8142.png" width=55% height=5%>
</p>

### Accuracy based on training size
<p float="left">
<img src="https://user-images.githubusercontent.com/20977650/222241483-ff20f89e-0ab3-41db-ab68-f01e967a7479.png" width=47% height=47%  hspace=10>
<img src="https://user-images.githubusercontent.com/20977650/222241762-bbb51660-e05c-42ac-ad41-054311275f0a.png" width=47% height=47%  hspace=10>
</p>

### Precision per label comparison
<p float="left">
<img src="https://user-images.githubusercontent.com/20977650/222242222-270007e8-7fff-4e82-8c65-ff261d4d80c6.png" width=47% height=47%  hspace=10>
<img src="https://user-images.githubusercontent.com/20977650/222242380-1ef8daa2-78bd-40df-85eb-fb5e10883faa.png" width=47% height=47%  hspace=10>
</p>


### Overall accuracy comparison of transformer models
<p float="left">
<img src="https://user-images.githubusercontent.com/20977650/222242519-083315b6-bac6-4a97-bf36-142a425539d9.png" width=47% height=47%  hspace=10 BORDER="0">
<img src="https://user-images.githubusercontent.com/20977650/222242600-9c6cd674-f851-43f2-a2cc-c23b45cfc000.png" width=47% height=47%  hspace=10 BORDER="0">
</p>

## Conclusions and further work
Simple Machine Learning models provide a pretty good performance for text classification tasks even considering that sentiment analysis is easier to do than similar language discrimination. The more complex MLP model does not necessarily provide better performance results. Our high accuracy figures also have to do with the type of data we used (curated DSLCC data and highly polar IMDB reviews). We discovered that contrary to common belief, removing stop words can actually hamper performance.
This shows that there is no "one-size-fits-all" recipe for supervised learning: each problem at hand requires careful feature design based on the nature of the task.
As further work, it would be worth evaluating the performance of Recurrent Neural Networks for our tasks, and to see if embeddings such as word2vec could help improve the performance.
