import os
import math
import numpy as np
from joblib import dump, load
from testutils import run_classification_experiment, get_random_acc

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from utils import split_data 

def test_rand_acc_balanced():
    y = np.array([1,1,2,2,3,3])
    assert get_random_acc(y) == 1.0/3.0

def test_rand_acc_imbalanced():
    y = np.array([1,3,3,3,3])
    assert get_random_acc(y) == 0.8
    
    
# write  a test case to check if model is successfully getting created or not?
def test_model_writing():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    x_train, x_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.3, shuffle=False)

    x_val, x_test, y_val, y_test = train_test_split(
        x_test,y_test, test_size=0.5, shuffle=False)

    gamma = 0.001
    classifier = svm.SVC(gamma=gamma)

    curr = "D:/IITJ/Semester-3/MLOps_HandsON/ML_Ops"
    output_model_file = curr + '/models/model_{}.joblib'.format(gamma)
    run_classification_experiment(classifier, x_train, y_train, x_val, y_val, gamma, output_model_file)

    assert os.path.isfile(output_model_file)


# write a test case to check fitting on training -- litmus test.
def test_small_data_overfit_checking():
    # taking 40% of data as sample data
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    targets = digits.target[:int(0.4*len(data))]
    data = data[:int(0.4*len(data))]

    x_train, x_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.2, shuffle=False)

    x_val, x_test, y_val, y_test = train_test_split(
        x_test,y_test, test_size=0.5, shuffle=False)

    gamma = 0.001
    classifier = svm.SVC(gamma=gamma)

    curr = "D:/IITJ/Semester-3/MLOps_HandsON/ML_Ops"
    output_model_file = curr + '/models/model_{}.joblib'.format(gamma)
    train_metrics = run_classification_experiment(classifier, x_train, y_train, x_val, y_val, gamma, output_model_file)

    assert train_metrics['acc']  > 0.8
    assert train_metrics['f1'] > 0.7

def try_split(sample, split, tsplit):
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    
    targets = digits.target[:sample]
    data = data[:sample]

    x_train, y_train, x_test, y_test, x_val, y_val = split_data(
        data, targets, split, tsplit)
    
    return x_train, y_train, x_test, y_test, x_val, y_val
    
def test_split():
    #Case 1: n = 100 samples
    split = (0.20,0.10)
    t_split = 0.3
    sample = 100
    x_train, y_train, x_test, y_test, x_val, y_val = try_split(sample, split, t_split)

    #print(len(x_train), len(x_test), len(x_val))
    total_len = len(x_train) + len(x_test) + len(x_val)
    total_len_label = len(y_train) + len(y_test) + len(y_val)
    assert len(x_train) == 70
    assert len(x_test) == 20
    assert len(x_val) == 10
    assert total_len == sample

    # assert for labels
    assert len(y_train) == 70
    assert len(y_test) == 20
    assert len(y_val) == 10

    # number of samples in train set and label set are same
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_val) == len(y_val)
    assert total_len == total_len_label

    #Case 2: n = 9 samples
    split = (0.20,0.10)
    t_split = 0.3
    sample = 9
    x_train, y_train, x_test, y_test, x_val, y_val = try_split(sample, split, t_split)

    total_len = len(x_train) + len(x_test) + len(x_val)
    total_len_label = len(y_train) + len(y_test) + len(y_val)
    assert len(x_train) == 6
    assert len(x_test) == 2
    assert len(x_val) == 1
    assert total_len == sample

    # assert for labels
    assert len(y_train) == 6
    assert len(y_test) == 2
    assert len(y_val) == 1

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_val) == len(y_val)
    assert total_len == total_len_label

def test_determine():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    targets = digits.target[:int(0.4*len(data))]
    data = data[:int(0.4*len(data))]

    x_train, x_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.2, shuffle=False)

    x_val, x_test, y_val, y_test = train_test_split(
        x_test,y_test, test_size=0.5, shuffle=False)

    gamma = 0.001
    classifier = svm.SVC(gamma=gamma)

    curr = "D:/IITJ/Semester-3/MLOps_HandsON/ML_Ops"
    output_model_file = curr + '/models/run-1.joblib'
    train_metrics_1 = run_classification_experiment(classifier, x_train, y_train, x_val, y_val, gamma, output_model_file)

    gamma = 0.001
    classifier = svm.SVC(gamma=gamma)
    output_model_file = curr + '/models/run-2.joblib'
    train_metrics_2= run_classification_experiment(classifier, x_train, y_train, x_val, y_val, gamma, output_model_file)
    
    assert train_metrics_1['acc']  == train_metrics_2['acc']
    assert train_metrics_1['f1'] == train_metrics_2['f1']

def test_model_corrupt():
    curr = "D:/IITJ/Semester-3/MLOps_HandsON/ML_Ops"
    output_model_file = curr + '/models/run-1.joblib'
    model = load(output_model_file)
    assert model