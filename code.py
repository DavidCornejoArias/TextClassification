# Importing packages
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
# Importing the data
Train = pd.read_excel('data.xlsx','sheet1')
trainDF = pd.DataFrame()
# training DF
trainDF['text'] = Train['text']
trainDF['label'] = Train['label']
trainDF['id'] = Train['ID_PRODUCT']
# splitting the dataframe
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
## label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
# Tranforming the text into number vectors within different techniques

# Count vector as feature
# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# Functions to run different models:
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)
# Modeling with different types of models and looking at the results
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("NB, Count Vectors: ", accuracy)
# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, Count Vectors: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, Count Vectors: ", accuracy)

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("NB, Count Vectors: ", accuracy)
# Linear classifier

# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, Count Vectors: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, Count Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, Count Vectors: ", accuracy)

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVM, N-Gram Vectors: ", accuracy)

# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print("RF, Count Vectors: ", accuracy)

# Here I run, the best model again and look at the specific metrics
classifier = ensemble.RandomForestClassifier().fit(xtrain_count, train_y)
predictd_proba = classifier.predict_proba(xvalid_count)
predictions = classifier.predict(xvalid_count)
metrics.accuracy_score(valid_y, predictions)
metrics.recall_score(valid_y, predictions, average='weighted')
metrics.precision_score(valid_y, predictions, average='weighted')
cm = confusion_matrix(valid_y, predictions)
# Creating a dataframe with different models to look at the variation in
# treshold, accuracy, recall and confusion matrix
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    model = classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    else:
        pass
    return model
# Models to run
modelNames = ['RandomForestClassifier','naive_bayes']
# Types of data for x
xTypes = [xtrain_count,xtrain_tfidf,xtrain_tfidf_ngram_chars]
validTypes = [xvalid_count, xvalid_tfidf, xtrain_tfidf_ngram_chars]
# validation data types
validTypesString = ['xvalid_count', 'xvalid_tfidf','xtrain_tfidf_ngram_chars']
# Creating objects where the information will be recorded
treshholdList = []
accuracyList = []
recallList = []
xtypeName = []
modelNamesList = []
precisionList = []
arrayList = []
confusionMatrixList = []
df = pd.DataFrame()
for model in modelNames:
    for x in range(0,len(xTypes)):
        if type(model) == str and model != 'naive_bayes':
            model = train_model(getattr(ensemble,model)(), xTypes[x], train_y, validTypes[x])
            predictd_proba = model.predict_proba(xvalid_count)
        elif type(model) == str and model == 'naive_bayes':
            model = train_model(getattr(naive_bayes,'MultinomialNB')(), xTypes[x], train_y, validTypes[x])
            predictd_proba = model.predict_proba(xvalid_count)            
        # array of values of treshold for each model
        array = [0.5, 0.6, 0.7, 0.8, 0.9,1]
        for i in array:
            arrayList.append(i)
            treshold = i
            treshholdList.append(treshold)
            predicted = (predictd_proba[:,1] >= treshold).astype('int')
            accuracy = metrics.accuracy_score(valid_y, predicted)
            accuracyList.append(accuracy)
            recall =  metrics.recall_score(valid_y, predicted)
            recallList.append(recall)
            precision = metrics.precision_score(valid_y, predicted)
            precisionList.append(precision)
            cm = confusion_matrix(valid_y, predicted)
            confusionMatrixList.append(cm)
            modelNamesList.append(str(model))
            xtypeName.append(validTypesString[x])
df['treshold']=arrayList
df['accuracy'] = accuracyList
df['precision'] = precisionList
df['recall'] = recallList
df['confusion'] = confusionMatrixList
df['model'] = modelNamesList
df['dataType'] = xtypeName
df.to_csv(r'results.csv')
