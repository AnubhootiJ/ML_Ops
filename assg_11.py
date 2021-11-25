
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocess, split_data, get_best, plot, split_data_shuffle

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

print("=============================\nClassifying Handwritten Digits")
print("=============================")


digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#
# gammas = [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05]
splits = [(0.1,0.1)]
train_split = [10,20,30,40,50,60,70,80,90,100]
curr = os.getcwd()

print("Now training...")
#d_ac = []
#s_ac = []

s_cols = ['Train Data (%)', 'gamma', 'SVMTestAcc', 'SVMValAcc', 'SVMF1Score']
svm_output = pd.DataFrame(data = [], columns=s_cols)
gammas = [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05]

d_cols = ['Train Data (%)', 'MaxDepth', 'DecTestAcc',  'DecValAcc', 'DecF1Score']
dt_output = pd.DataFrame(data = [], columns=d_cols)
depths = [6,8,10,12,14,16]


def train_dec(x_train, y_train, x_val, y_val, x_test, y_test, depth, cmd=False, td=None):
    dec = DecisionTreeClassifier(max_depth=depth)
    dec.fit(x_train, y_train)
    #print(dec.get_depth())
    t_ac = dec.score(x_test, y_test)
    val_ac = dec.score(x_val, y_val)
    predicted = dec.predict(x_test)
    f1 = metrics.f1_score(y_pred=predicted,y_true=y_test, average='macro')
    if cmd:
        cm = metrics.confusion_matrix(predicted, y_test, labels  = [0,1,2,3,4,5,6,7,8,9])
        disp = metrics.ConfusionMatrixDisplay(cm)
        ttl = 'Confusion Matrix for ' + str(td) + '% training data'
        disp.plot()
        plt.title(ttl)
        pth = 'results/dec_cm/' + str(td) + '.jpg'
        plt.savefig(pth)
        #plt.show()
    return t_ac, val_ac, f1

def train_svm(x_train, y_train, x_val, y_val, x_test, y_test, gamma, cmd=False, td = None):
    clf = svm.SVC(gamma=gamma)
    clf.fit(x_train, y_train)
    st_ac = clf.score(x_test, y_test)
    sval_ac = clf.score(x_val, y_val)
    predicted = clf.predict(x_test)
    sf1 = metrics.f1_score(y_pred=predicted,y_true=y_test, average='macro')
    if cmd:
        cm = metrics.confusion_matrix(predicted, y_test, labels  = [0,1,2,3,4,5,6,7,8,9])
        disp = metrics.ConfusionMatrixDisplay(cm)
        ttl = 'Confusion Matrix for ' + str(td) + '% training data'
        disp.plot()
        plt.title(ttl)
        pth = 'results/svm_cm/' + str(td) + '.jpg'
        plt.savefig(pth)
        #plt.show()
    return st_ac, sval_ac, sf1

data = preprocess(digits.images, 32)
for split in splits:
    tsplit = split[0] + split[1]
    x_train, y_train, x_test, y_test, x_val, y_val = split_data_shuffle(
        data, digits.target, split, tsplit)
    
    for gamma in gammas:
        for tr in train_split:
            # get tr% train data
            sp = int(tr/100 * len(x_train))
            n_train = x_train[:sp]
            n_ytrain = y_train[:sp]
            # training SVM
            if gamma == 0.001:
                st_ac, sval_ac, sf1 = train_svm(n_train, n_ytrain, x_val, y_val, x_test, y_test, gamma, True, tr)
            else:
                
                st_ac, sval_ac, sf1 = train_svm(n_train, n_ytrain, x_val, y_val, x_test, y_test, gamma)
            #s_ac.append(sval_ac)
            out = pd.DataFrame(data = [[tr, gamma, st_ac, sval_ac, sf1]],
            columns = s_cols)
            #print(out)
            svm_output = svm_output.append(out, ignore_index=True)
    
    for depth in depths:
        for tr in train_split:
            # get tr% train data
            sp = int(tr/100 * len(x_train))
            n_train = x_train[:sp]
            n_ytrain = y_train[:sp]
            if depth == 12:
                t_ac, val_ac, f1 = train_dec(n_train, n_ytrain, x_val, y_val, x_test, y_test, depth, True, tr)
            else:
                t_ac, val_ac, f1 = train_dec(n_train, n_ytrain, x_val, y_val, x_test, y_test, depth)
            #d_ac.append(val_ac)
            out = pd.DataFrame(data = [[tr, depth, t_ac, val_ac, f1]],
            columns = d_cols)
            #print(out)
            dt_output = dt_output.append(out, ignore_index=True)

print("Stats for SVM Training - ")
print(svm_output)
svm_output.to_csv("D:\IITJ\Semester-3\MLOps_HandsON\ML_Ops\svm_output.csv")

print("Stats for Decision Tree Training - ")
print(dt_output)
dt_output.to_csv("D:\IITJ\Semester-3\MLOps_HandsON\ML_Ops\dt_output.csv")


