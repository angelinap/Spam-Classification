#Angelina Poole
#COEN 140
#Homework 4 Bonus

import numpy as np
import math
from scipy import stats

# from sklearn import preprocessing
# import sklearn.metrics as sk_metrics
# from sklearn.linear_model import LogisticRegression

##################################################################
def standardize(xdata):
    return stats.zscore(xdata, axis = 1)

##################################################################
def transform(elt):
    return math.log(elt + 0.1)


def log_transform(xdata):
    for i, instance in enumerate(xdata):
        xdata[i] = map(transform, instance)
    return xdata

##################################################################
def binarize(elt):
    if (elt > 0):
        elt = 1
    else:
        elt = 0
    return elt

def bin_transform(xdata):
    for i, instance in enumerate(xdata):
        xdata[i] = map(binarize, instance)
    return xdata

##################################################################
# def init_regression_model(xdata, ydata):
#     regression_model = LogisticRegression()
#     regression_model.fit(xdata, ydata)
#     return regression_model
#
# def predict_results(testdata, regression_model):
#     return regression_model.predict(testdata)
##################################################################

#s(r) = 1/1+e^-r where r is w^T xi
def sigmoid(scores):
    return 1/(1+np.exp(-scores))

def sigmoid_func(matrix, weights):
    weights_t = np.transpose(weights)
    dot_prod = np.dot(matrix, weights)
    exp_val = math.e ** (-1 * dot_prod)
    denom = 1.0 + exp_val
    result = 1/denom
    return result

def regression_closed_testing(w, xdata):
    print "in regression closed testing"
    print "W SHAPE"
    print np.shape(w)
    w_t = np.transpose(w)
    xdata_t = np.transpose(xdata)
    ynew = np.dot(w_t, xdata_t)
    print "Y NEW SHAPE"
    print np.shape(ynew)
    return np.transpose(ynew)

def log_likelihood(xdata, ydata, weights):
    scores = np.dot(ydata, weights)
    ll = np.sum(xdata*scores - np.log(1 + np.exp(scores)))
    return ll

#w^t+1 = wt + alpha*xT(y-s(xw^t))
def logistic_regression_gradient_descent(xdata, ydata, alpha):
    it = 0
    # w_curr = w^t
    w_curr = np.random.uniform(0, 1, (58, 1))
    xdata_t = np.transpose(xdata)
    middle_part = np.multiply(alpha, xdata_t)
    right_part = np.subtract(ydata, sigmoid_func(xdata, w_curr))
    total_right_part = np.dot(middle_part, right_part)
    w_next = np.add(w_curr, total_right_part)
    diff = 1
    it = 1
    while(abs(diff) > .0001):
        w_curr = w_next
        middle_part = np.multiply(alpha, xdata_t)
        right_part = np.subtract(ydata, sigmoid_func(xdata, w_curr))
        total_right_part = np.dot(middle_part, right_part)
        w_next = np.add(w_curr, total_right_part)
        diff = np.amin(np.subtract(w_next, w_curr), 0)
        it += 1
    return w_next

def classificationAccuracy(ynew, ytrue):
    rows = ynew.shape[0]
    accuracy = 0.0
    for i in range(0, rows):
        if (ytrue[i] == ynew[i]):
            accuracy += 1
    print accuracy
    # print "HERE"
    # print accuracy/ynew.shape[0]
    return accuracy/rows

##################################################################
#Put the training data into a list
trd = []
with open('spambase/spam-train.txt','r') as input_file:
    #input_file.next()
    for line in input_file:
        row = line.split(',')
        for i in range(0, 58):
            row[i] = float(row[i])
        #row[57] = line.strip('\n')
        trd.append(row)

# print "PRINT TRAIN"
# print trd[0]
# #print "original train list"
# #print trd
#
# print "rows including the row of col names"
# train_rows = len(trd)
# print train_rows
#
# print "columns"
# train_cols = len(trd[0])
# print train_cols

#Put the training list into a numpy array
# print "original train numpy with bool labels at the last col"
train_numpy = np.asarray(trd)
# print train_numpy
# print "train numpy without labels"
train_numpy_new = train_numpy[:,0:57]
# print train_numpy_new
# print np.shape(train_numpy_new)
# print train_numpy_new[0][56]
# print np.shape(train_numpy_new)

#want to get the last column because those are boolean values to say whether it is spam or not
yvalue_train = train_numpy[:,57]
yvalue_train = yvalue_train.reshape(3065, 1)
# print "yvalue_train"
# print yvalue_train
##################################################################
#Put the test data into a list
tstd = []
with open('spambase/spam-test.txt','r') as input_file:
    #input_file.next()
    for line in input_file:
        row = line.split(',')
        for i in range(0, 58):
            row[i] = float(row[i])
        #row[57] = line.strip('\n')
        tstd.append(row)

# print "PRINT TEST"
# print tstd[0]
# #print "original train list"
# #print tstd
#
# print "rows including the row of col names"
# test_rows = len(tstd)
# print test_rows
#
# print "columns"
# test_cols = len(tstd[0])
# print test_cols

# #Put the testing list into a numpy array
# print "original test numpy with bool labels at the last col"
test_numpy = np.asarray(tstd)
# print test_numpy
# print test_numpy[0][57]
# print "test numpy without labels"
test_numpy_new = test_numpy[:,0:57]
# print test_numpy_new
# print np.shape(test_numpy)
#want to get the last column because those are boolean values to say whether it is spam or not
yvalue_test = test_numpy[:,57]
yvalue_test = yvalue_test.reshape(1536, 1)
# print "yvalue_test"
# print yvalue_test
##################################################################
#Preprocessing
ones_train = np.ones((3065,1))
ones_test = np.ones((1536, 1))

#1) Standardize the columns so they all have mean 0 and unit variance.
#Which one?
standardized_train_numpy = standardize(train_numpy_new)
# print "standardized train"
# print standardized_train_numpy
# print "sklearn standardize check"
# print preprocessing.scale(train_numpy_new)
standardized_test_numpy = standardize(test_numpy_new)
# print "standardized test"
# print standardized_test_numpy

standardized_train_numpy = np.append(standardized_train_numpy, ones_train, axis = 1)
# print "ones col added"
# print "standardized train numpy"
# print standardized_train_numpy

standardized_test_numpy = np.append(standardized_test_numpy, ones_test, axis = 1)
# print "ones col added"
# print "transform test numpy"
# print standardized_test_numpy

##################################################################
#2) Transform the features using log(xij + 0.1)
train_numpy_new1 = train_numpy_new
test_numpy_new1 = test_numpy_new

transform_train_numpy = log_transform(train_numpy_new1)
# print "transformed training"
# print transform_train_numpy
#
# print "train numpy22"
# print train_numpy_new

transform_test_numpy = log_transform(test_numpy_new1)
# print "transformed test"
# print transform_test_numpy

transform_train_numpy = np.append(transform_train_numpy, ones_train, axis = 1)
# print "ones col added"
# print "transform train numpy"
# print transform_train_numpy

transform_test_numpy = np.append(transform_test_numpy, ones_test, axis = 1)
# print "ones col added"
# print "transform test numpy"
# print transform_test_numpy
##################################################################
#3) Binarize the features using I(xij > 0)
bin_train_numpy = bin_transform(train_numpy_new1)
# print "binarized train"
# print bin_train_numpy
#
# print "binarized sklearn check training"
# bin_train_numpy2 = preprocessing.binarize(train_numpy_new)
# print bin_train_numpy2
#
# print "equal?"
# print np.array_equal(bin_train_numpy, bin_train_numpy2)
bin_test_numpy = bin_transform(test_numpy_new1)
# print "binarized test"
# print np.shape(bin_test_numpy)
#
# print "binarized sklearn check test"
# bin_test_numpy2 = preprocessing.binarize(test_numpy_new)
# print bin_train_numpy2
#
# print "equal?"
# print np.array_equal(bin_test_numpy, bin_test_numpy2)

bin_train_numpy = np.append(bin_train_numpy, ones_train, axis = 1)
# print "ones col added"
# print "bin train numpy"
# print bin_train_numpy

bin_test_numpy = np.append(bin_test_numpy, ones_test, axis = 1)
# print "ones col added"
# print "bin test numpy"
# print bin_test_numpy
##################################################################
#TRAINING Logistic Regression Gradient Descent
print "LOGISTIC REGRESSION GRADIENT DESCENT"

weight_lrg_train1 = logistic_regression_gradient_descent(standardized_train_numpy, yvalue_train, 0.00001)
weight_lrg_train2 = logistic_regression_gradient_descent(transform_train_numpy, yvalue_train, 0.00001)
weight_lrg_train3 = logistic_regression_gradient_descent(bin_train_numpy, yvalue_train, 0.00001)
print "1) Weights for Standardized Data"
print weight_lrg_train1
print "2) Weights for Log Transformed Data"
print weight_lrg_train2
print "3) Weights for Binarized Data"
print weight_lrg_train3
##################################################################
#TESTING
#ERROR RATES for Logistic Regression Gradient Descent
pval_logistic_regression_train1 = regression_closed_testing(weight_lrg_train1, standardized_train_numpy)
pval_logistic_regression_train2 = regression_closed_testing(weight_lrg_train2, transform_train_numpy)
pval_logistic_regression_train3 = regression_closed_testing(weight_lrg_train3, bin_train_numpy)

predtrain1 = np.where(sigmoid(pval_logistic_regression_train1) > 0.5, 1, 0)
predtrain2 = np.where(sigmoid(pval_logistic_regression_train2) > 0.5, 1, 0)
predtrain3 = np.where(sigmoid(pval_logistic_regression_train3) > 0.5, 1, 0)

pval_logistic_regression_test1 = regression_closed_testing(weight_lrg_train1, standardized_test_numpy)
pval_logistic_regression_test2 = regression_closed_testing(weight_lrg_train2, transform_test_numpy)
pval_logistic_regression_test3 = regression_closed_testing(weight_lrg_train3, bin_test_numpy)

print "Predtest1"
predtest1 = np.where(sigmoid(pval_logistic_regression_test1) > 0.5, 1, 0)
print "Predtest2"
predtest2 = np.where(sigmoid(pval_logistic_regression_test2) > 0.5, 1, 0)
print "Predtest3"
predtest3 = np.where(sigmoid(pval_logistic_regression_test3) > 0.5, 1, 0)

accuracy_train1 = classificationAccuracy(predtrain1, yvalue_train)
accuracy_test1 = classificationAccuracy(predtest1, yvalue_test)

accuracy_train2 = classificationAccuracy(predtrain2, yvalue_train)
accuracy_test2 = classificationAccuracy(predtest2, yvalue_test)

accuracy_train3 = classificationAccuracy(predtrain3, yvalue_train)
accuracy_test3 = classificationAccuracy(predtest3, yvalue_test)

print "##################################################################"
print "Error Rates for Logistic Regression Closed Form:"
print "1) Standardized Data"
print "Error Rate Train is %s" %(1-accuracy_train1)
print "Error Rate Test is %s" %(1-accuracy_test1)
print "2) Log Transformed Data"
print "Error Rate Train is %s" %(1-accuracy_train2)
print "Error Rate Test is %s" %(1-accuracy_test2)
print "3) Binarized Data"
print "Error Rate Train is %s" %(1-accuracy_train3)
print "Error Rate Test is %s" %(1-accuracy_test3)
print "##################################################################"

#REGULAR LINEAR REGRESSION
# log_regression_model = init_regression_model(transform_train_numpy, yvalue_train)
# my_results = log_regression_model.predict(transform_train_numpy)
# print sk_metrics.mean_squared_error(yvalue_train, my_results)
