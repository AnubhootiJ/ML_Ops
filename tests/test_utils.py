import os
import numpy as np
from testutils import run_classification_experiment, get_random_acc

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

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