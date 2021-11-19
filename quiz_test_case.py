import os
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
targets = digits.target

def get_test_case(target):
    i = 0
    while targets[i] != target:
        i+=1
    #print(data[i].reshape(-1,1).shape)
    return data[i].reshape(1,-1)

def get_model():
    path = "D:/IITJ/Semester-3/MLOps_HandsON/ML_Ops/models/model_0.01.joblib"
    dec_path = "D:/IITJ/Semester-3/MLOps_HandsON/ML_Ops/models/model_8_(0.2, 0.1)_0.01.joblib"
    svm_clf = load(path)
    dec_clf = load(dec_path)

    print("Model Loaded")
    return svm_clf, dec_clf


def get_test_accuracy(target):
    dt = []
    for i,tgt in enumerate(targets):
        if tgt == target:
            dt.append(data[i])
    dt = np.array(dt)
    return dt

def get_mean(svm_pred, dec_pred, target):
    tgt = np.full(len(svm_pred), target)
    svm_acc = np.sum(np.equal(svm_pred, tgt))
    svm_acc = svm_acc/len(svm_pred)

    tgt = np.full(len(dec_pred), target)
    dec_acc = np.sum(np.equal(dec_pred, tgt))
    dec_acc = dec_acc/len(dec_pred)

    return svm_acc, dec_acc

def accuracy_class(svm_clf, dec_clf, target):
    dt = get_test_accuracy(target)
    svm_pred = svm_clf.predict(dt)
    dec_pred = dec_clf.predict(dt)
    svm_acc, dec_acc = get_mean(svm_pred, dec_pred, target)
    return svm_acc, dec_acc

