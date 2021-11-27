import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump

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

gammas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
Cs = [0.1,0.5,1,1.5]

fdf = pd.DataFrame(columns = ['Gamma', 'C', 'R1_Train', 'R1_Dev', 'R1_Test',
'R2_Train', 'R2_Dev', 'R2_Test',
'R3_Train', 'R3_Dev', 'R3_Test'])
R1 = []

x_train, y_train, x_test, y_test, x_val, y_val = split_data(
            data, digits.target, (0.15, 0.15), 0.7)
for gamma in gammas:
    for c in Cs:
        clf = svm.SVC(gamma=gamma, C=c)
        clf.fit(x_train, y_train)
        t_ac = clf.score(x_train, y_train)
        val_ac = clf.score(x_val, y_val)
        test_ac = clf.score(x_test, y_test)
        vals = [t_ac, val_ac, test_ac]
        R1.append([t_ac, val_ac, test_ac])
        
        mean = np.mean([t_ac, val_ac, test_ac])
        if mean>0.8:
            pth = 'Model_' + str(gamma) + "_" + str(c) + '_.joblib'
            clf.dump("D:/IITJ/Semester-3/MLOps_HandsON/ML_Ops/Model/"+pth)


        

