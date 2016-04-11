# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 16:13:40 2016

@author: zck
"""

import arff, os, sys
import numpy as np
from sklearn import svm

def load_feature(filename):
    print os.path.isfile(filename)
    data_arff = arff.load(open(filename, 'rb'))
    N = len( data_arff['data'] )
    M = len( data_arff['attributes'] ) - 2
    x = np.zeros([N,M])
    y = []
    for i in xrange(N):
        for j in xrange(M):
            x[i,j] = data_arff['data'][i][j+1]
            
    for i in xrange(N):
        y.append(data_arff['data'][i][M+1])
        
    return [x, y]

def run_train_val(xTr, yTr, xVal, yVal, clf, method_name):
    try:
        clf.fit(xTr, yTr)
#        acc = clf.score(xTr, yTr)
#        print('{0} method, Train accuracy is {1}'.format(method_name, acc) )
        acc = clf.score(xVal, yVal)
        print('{0} method, Val accuracy is {1}'.format(method_name, acc) )
    except:
        pass


"""
http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis
"""
def run_discriminant_analysis(xTr, yTr, xVal, yVal):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    run_train_val(xTr, yTr, xVal, yVal, LinearDiscriminantAnalysis(), 'LDA')
    run_train_val(xTr, yTr, xVal, yVal, QuadraticDiscriminantAnalysis(), 'QDA')

"""
http://scikit-learn.org/stable/modules/tree.html
"""    
def run_tree(xTr, yTr, xVal, yVal):
    from sklearn import tree
    run_train_val(xTr, yTr, xVal, yVal, tree.DecisionTreeClassifier(), 'DecisionTree')

"""
http://scikit-learn.org/stable/modules/naive_bayes.html
"""
def run_naive_bayes(xTr, yTr, xVal, yVal):
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    run_train_val(xTr, yTr, xVal, yVal, GaussianNB(), 'GaussianNB')
    run_train_val(xTr, yTr, xVal, yVal, MultinomialNB(), 'MultinomialNB')
    run_train_val(xTr, yTr, xVal, yVal, BernoulliNB(), 'BernoulliNB')

"""
http://scikit-learn.org/stable/modules/linear_model.html
"""
def run_linear_model(xTr, yTr, xVal, yVal):
    from sklearn import linear_model
    run_train_val(xTr, yTr, xVal, yVal, linear_model.SGDClassifier(n_iter=100, n_jobs=-1),
                  'linear_model hinge l2')
    run_train_val(xTr, yTr, xVal, yVal, linear_model.SGDClassifier(penalty='l1',n_iter=100, n_jobs=-1),
                  'linear_model hinge l1')
    run_train_val(xTr, yTr, xVal, yVal, linear_model.SGDClassifier(penalty='elasticnet',n_iter=100, n_jobs=-1),
                  'linear_model hinge elasticnet')

"""
http://scikit-learn.org/stable/modules/neighbors.html
"""   
def run_neighbors(xTr, yTr, xVal, yVal):
    from sklearn import neighbors
    for i in xrange(2, 53, 10):
        run_train_val(xTr, yTr, xVal, yVal, 
                      neighbors.KNeighborsClassifier(n_neighbors=i, algorithm='ball_tree'), '{0}KNN'.format(i))
    
"""
"""
def run_svm(xTr, yTr, xVal, yVal):
    run_train_val(xTr, yTr, xVal, yVal, svm.LinearSVC(), 'linear svm')
    run_train_val(xTr, yTr, xVal, yVal, svm.SVC(), 'rbf svm')
#    run_train_val(xTr, yTr, xVal, yVal, svm.SVC(kernel='poly', degree=3), 'poly svm')
    run_train_val(xTr, yTr, xVal, yVal, svm.SVC(kernel='sigmoid'), 'sigmoid svm')
    


if __name__ == '__main__':
    train_filename = '..\\..\\CCPR-data\\feature\\train\\train_video.arff'
    val_filename = '..\\..\\CCPR-data\\feature\\val\\val_video.arff'
    
#    train_filename = '..\\..\\CCPR-data\\feature\\train\\train_audio.arff'
#    val_filename = '..\\..\\CCPR-data\\feature\\val\\val_audio.arff'
    
    [xTr, yTr]    = load_feature(train_filename)
    [xVal, yVal]  = load_feature(val_filename  )
    
    run_neighbors(xTr, yTr, xVal, yVal)
    run_linear_model(xTr, yTr, xVal, yVal)
    run_tree(xTr, yTr, xVal, yVal)
    run_discriminant_analysis(xTr, yTr, xVal, yVal)
    run_naive_bayes(xTr, yTr, xVal, yVal)
    run_svm(xTr, yTr, xVal, yVal)    
#    run_train_val(xTr, yTr, xVal, yVal, svm.SVC(decision_function_shape='ovo'), 'svm test')
    
    