import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def split_data(data, target, split, tsplit):
    v_split = split[0]
    t_split = split[1]
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, train_size=1-tsplit, test_size=tsplit, shuffle=False)

    x_val, x_test, y_val, y_test = train_test_split(
        x_test,y_test, test_size=v_split/(t_split+v_split), shuffle=False)
    #print("\nNumber of samples in train:val:test = {}:{}:{}".format(len(x_train), len(x_val), len(x_test)))

    return x_train, y_train, x_test, y_test, x_val, y_val


digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

lrs = [1e-4, 1e-3, 1e-2, 0.1, 1, 10]

x_train, y_train, x_test, y_test, x_val, y_val = split_data(
        data, digits.target, (0.15, 0.15), 0.7)
for lr in lrs:
    clf =  MLPClassifier(learning_rate_init = lr)
    clf.fit(x_train, y_train)
    t_ac = clf.score(x_train, y_train)
    val_ac = clf.score(x_val, y_val)
    test_ac = clf.score(x_test, y_test)
    vals = [t_ac, val_ac, test_ac]
    print(lr, "= ", t_ac, val_ac, test_ac)

def train_mlp(lr):
    x_train, y_train, x_test, y_test, x_val, y_val = split_data(
        data, digits.target, (0.15, 0.15), 0.7)
    x_train = x_train[:int(0.1*len(x_train))]
    y_train = y_train[:int(0.1*len(y_train))]
    clf =  MLPClassifier(learning_rate_init = lr)
    clf.fit(x_train, y_train)
    t_ac = clf.score(x_train, y_train)
    
    return t_ac