import collections
import csv
import math
import re
import pickle

import numpy as np
import sklearn.metrics
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

###########################################################################################
#   PREPROCESSING
###########################################################################################
def open_file(file_name):
    instances = []
    labels = []
    data = csv.reader(open(file_name))
    next(data)  # Skip header row
    for line in data:
        instances.append(line[0:3])
        if len(line) == 4: 
            labels.append(int(line[3]))
    return (instances, labels)

def count_vectorize(instances):
    vectorizer = CountVectorizer(analyzer = 'word', stop_words = stopwords.words('english'))
    corpus = []
    for i in instances:
        # appends the phrase of each instance into the corpus
        corpus.append(i[2])
    corpus_fitted = vectorizer.fit(corpus)
    corpus_transformed = corpus_fitted.transform(corpus)

    print('Count Vectorizer Vocab Length: ' + str(len(vectorizer.vocabulary_)))
    return corpus_transformed

def tfid_vectorize_train(instances):    
    # Store all phrases from instances into corpus list
    corpus = []    
    for i in instances:        
        corpus.append(i[2])

    # Initialize the 
    vectorizer = TfidfVectorizer(analyzer='word', preprocessor=None, stop_words=stopwords.words('english'), lowercase = True, min_df=5)
    corpus_fitted = vectorizer.fit(corpus)
    pickle.dump(corpus_fitted, open('tfidf1.pkl', 'wb'))

    corpus_transformed = corpus_fitted.transform(corpus)

    print('tfidf Vectorizer test Vocab Length: ' + str(len(vectorizer.vocabulary_)))
    return corpus_transformed

def tfid_vectorize_test(instances):
    corpus = []
    for i in instances:
        # appends the phrase of each instance into the corpus
        corpus.append(i[2])

    train_data = pickle.load(open('tfidf1.pkl', 'rb'))
    vectorizer = TfidfVectorizer(analyzer='word', preprocessor=None, stop_words=stopwords.words('english'), lowercase=True, vocabulary=train_data.vocabulary_)

    corpus_transformed = vectorizer.fit_transform(corpus)
    return corpus_transformed


###########################################################################################
#   CLASSIFIERS
###########################################################################################
def linear_regression(train_instances, train_labels, test_instances):    
    print('training model')
    vanilla_linear_regression = LinearRegression(fit_intercept=True, normalize=True).fit(train_instances, train_labels)
    print('finished training model')
    predicted_labels = vanilla_linear_regression.predict(test_instances)
    rounded = []
    for i in predicted_labels: 
        rounded.append(math.trunc(i))
    return rounded 

def naive_bayes(train_instances, train_labels, test_instances):
    nb = MultinomialNB()
    nb.fit(train_instances, train_labels)
    predicted_labels = nb.predict(test_instances)
    return predicted_labels

def svm(train_instances, train_labels, test_instances):
    svm = LinearSVC(random_state=0, tol=1e-5, max_iter=4000, C=5)
    svm_fit = svm.fit(train_instances, train_labels)
    prediction = svm_fit.predict(test_instances)
    return prediction

def perceptron(train_instances, train_labels, test_instances):
    percep = Perceptron(tol=1e-3, random_state=0)
    percep.fit(train_instances, train_labels)
    prediction = percep.predict(test_instances)
    return prediction


###########################################################################################
#   TESTING/VALIDATION
###########################################################################################
def printPerformance(model_name, expected_labels, predicted_labels):
    labels = [0, 1, 2, 3, 4]
    conf_matrix = sklearn.metrics.confusion_matrix(expected_labels, predicted_labels, labels = labels)
    accuracy = sklearn.metrics.accuracy_score(expected_labels, predicted_labels)
    precision = sklearn.metrics.precision_score(expected_labels, predicted_labels, labels=labels, average=None)
    recall = sklearn.metrics.recall_score(expected_labels, predicted_labels, labels=labels, average=None)
    
    print('='*60)
    print(model_name)
    print('='*60)
    print('CONFUSION MATRIX:')
    print(conf_matrix)
    print('ACCURACY: ' + str(accuracy) + ' Ava')
    print('RECALL: ' + str(recall))
    print('PRECISION: ' + str(precision) + '\n')   

def cross_validation(instance, labels):
    clf = LinearSVC(random_state=0, tol=1e-5, C=5, max_iter=4000)
    cv_result = cross_validate(
        clf, instance, labels, cv=10, return_train_score=True)
    test_score = cv_result.get('test_score')
    train_score = cv_result.get('train_score')
    print('10 folds, test score: ', test_score, "train score: ", train_score)

def compile_output_file(instances, predicted_labels):
    with open('RoyShinganeYu_predictionsLOOKATTHISONE.csv', mode='w') as output:
        file_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Header row
        file_writer.writerow(['PhraseID', 'Sentiment'])

        # Write individual instance sentiments
        for i in range(len(predicted_labels)):
            file_writer.writerow([instances[i][0], predicted_labels[i]])


###########################################################################################
#   PROGRAM STARTS HERE
###########################################################################################
if __name__ == '__main__':
    # Load training data
    train_instances, train_labels = open_file('train.csv')    

    # Number of training instances
    print('Number of training instances: ' + str(len(train_instances)))

    # TFIDF Vectorize training phrases
    tfid_vector_matrix = tfid_vectorize_train(train_instances).astype(np.float64)

    # Count Vectorize training phrases
    count_vector_matrix = count_vectorize(train_instances).astype(np.float64)

    # Load test data
    # Since this is using the same function as the train set, it will output an empty labels list. We can ignore this. 
    file_test_instances, throwaway = open_file('testset_1.csv')

    # TFIDF Vectorize test instances using the training vocabulary. 
    test_instances = tfid_vectorize_test(file_test_instances)
       
    # Call vanilla regression
    # lin_reg_predicted_labels = linear_regression(tfid_vector_matrix, train_labels, test_instances)
    # lin_reg_predicted_labels_count = linear_regression(count_vector_matrix, train_labels, test_instances)

    # printPerformance('Vanilla Linear Regression + tfid', train_labels, lin_reg_predicted_labels)
    # printPerformance('Vanilla Linear Regression + count', train_labels, lin_reg_predicted_labels_count)

    # Call Naive Bayes classifier
    # naive_bayes_predicted_labels = naive_bayes(tfid_vector_matrix, train_labels, test_instances)
    # naive_bayes_predicted_labels_count = naive_bayes(count_vector_matrix, train_labels, test_instances)

    # printPerformance('Naive Bayes + tfid', train_labels, naive_bayes_predicted_labels)
    # printPerformance('Naive Bayes + count', train_labels, naive_bayes_predicted_labels_count)

    # Call SVM classifier 
    svm_predicted_labels = svm(tfid_vector_matrix, train_labels, test_instances)
    # svm_predicted_labels_count = svm(count_vector_matrix, train_labels, test_instances)
# 
    # printPerformance('SVM + tfid', train_labels, svm_predicted_labels)
    # printPerformance('SVM + count', train_labels, svm_predicted_labels_count)

    # Call Perceptron classifier
    # percep_predicted_labels = perceptron(tfid_vector_matrix, train_labels, test_instances)
    # percep_predicted_labels_count = perceptron(count_vector_matrix, train_labels, test_instances)

    # printPerformance('Perceptron + tfid', train_labels, percep_predicted_labels)
    # printPerformance('Perceptron + count', train_labels, percep_predicted_labels_count)


    # # Voting    
    # svm_1 = LinearSVC(random_state=0, tol=1e-5, max_iter=10000, C=5)
    # svm_2 = LinearSVC(random_state=0, tol=1e-5, max_iter=10000, C=1)

    # voting_clf = VotingClassifier( estimators=[('svm1', svm_1), ('svm2', svm_2)], voting='soft')
    # voted_predicted_labels = voting_clf.fit(tfid_vector_matrix, train_labels).predict(tfid_vector_matrix)
    
    # cv_result = cross_validate(voting_clf, tfid_vector_matrix, train_labels, cv=10, return_train_score=True)
    # test_score= cv_result.get('test_score')
    # train_score= cv_result.get('train_score')
    # print('10 folds, test score: ', test_score, "train score: ", train_score)

    # PRINTING OUT FILE 
    # Change what output_label uses for file output by changing the predicted labels set
    output_labels = svm_predicted_labels
    compile_output_file(file_test_instances, output_labels)

    # # Cross validation
    # cross_validation(tfid_vector_matrix, train_labels)
    # cross_validation(count_vector_matrix, train_labels)
