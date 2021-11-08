
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
splits = [(0.15,0.15)]
resolution = [32]
curr = os.getcwd()

print("Now training...")
#candidates = []
i=1
#fig = plt.figure()
dec_mean = 0
dec_sd = 0
d_ac = []

svm_mean = 0
svm_sd = 0
s_ac = []
cols = ['SplitRun',  'DecTrainAcc',  'DecValAcc', 'DecF1Score', 'SVMTrainAcc', 'SVMValAcc', 'SVMF1Score']
output = pd.DataFrame(data = [], columns=cols)
#print("Val:Test  Dec Train Acc  Dec Val Acc Dec f1 Score SVM Train Acc SVM Val Acc SVM f1 Score")



def train_dec(x_train, y_train, x_val, y_val, x_test, y_test):
    dec = DecisionTreeClassifier()
    dec.fit(x_train, y_train)
    t_ac = dec.score(x_train, y_train)
    val_ac = dec.score(x_val, y_val)
    predicted = dec.predict(x_test)
    f1 = metrics.f1_score(y_pred=predicted,y_true=y_test, average='macro')
    return t_ac, val_ac, f1

def train_svm(x_train, y_train, x_val, y_val, x_test, y_test):
    clf = svm.SVC(gamma=0.001)
    clf.fit(x_train, y_train)
    st_ac = clf.score(x_train, y_train)
    sval_ac = clf.score(x_val, y_val)
    predicted = clf.predict(x_test)
    sf1 = metrics.f1_score(y_pred=predicted,y_true=y_test, average='macro')
    return st_ac, sval_ac, sf1

data = preprocess(digits.images, 32)
for split in splits:
    for i in range(5):
        tsplit = split[0] + split[1]
        x_train, y_train, x_test, y_test, x_val, y_val = split_data_shuffle(
            data, digits.target, split, tsplit)
        
        # training decision tree
        t_ac, val_ac, f1 = train_dec(x_train, y_train, x_val, y_val, x_test, y_test)
        dec_mean+=val_ac
        d_ac.append(val_ac)

        # training SVM
        st_ac, sval_ac, sf1 = train_svm(x_train, y_train, x_val, y_val, x_test, y_test)
        svm_mean+=sval_ac
        s_ac.append(sval_ac)

        out = pd.DataFrame(data = [[i+1, t_ac, val_ac, f1, st_ac, sval_ac, sf1]],
        columns = cols)
        #print(out)
        output = output.append(out, ignore_index=True)

print(output)
print("Mean accuracy for decision tree = {:.2f}".format(dec_mean/5) )     
print("Mean accuracy for SVM = {:.2f}".format(svm_mean/5) )   

for i in range(len(d_ac)):
    dec_sd += (d_ac[i]-dec_mean)**2
    svm_sd += (s_ac[i]-svm_mean)**2

dec_sd = (dec_sd/5)**(1/2)
svm_sd = (svm_sd/5)**(1/2)

print("SD accuracy for decision tree = {:.2f}".format(dec_sd/5) )     
print("SD accuracy for SVM = {:.2f}".format(svm_sd/5) )   























"""
import os
import matplotlib.pyplot as plt
from utils import preprocess, split_data, get_best, plot

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
splits = [(0.15,0.15), (0.20,0.10), (0.1, 0.1), (0.1, 0.05), (0.05, 0.05)]
resolution = [32]
curr = os.getcwd()

print("Now training...")
#candidates = []
i=1
#fig = plt.figure()
print("Val:Test  Dec Train Acc  Dec Val Acc Dec f1 Score SVM Train Acc SVM Val Acc SVM f1 Score")
for res in resolution:
    data = preprocess(digits.images, res)
    for split in splits:
        tsplit = split[0] + split[1]
        x_train, y_train, x_test, y_test, x_val, y_val = split_data(
            data, digits.target, split, tsplit)
        vaccs = []
        taccs = []

        SVMvaccs = []
        SVMtaccs = []
        
        #for gamma in gammas:

        dec = DecisionTreeClassifier(random_state = 0)
        dec.fit(x_train, y_train)
        t_ac = dec.score(x_train, y_train)
        val_ac = dec.score(x_val, y_val)
        predicted = dec.predict(x_test)
        f1 = metrics.f1_score(y_pred=predicted,y_true=y_test, average='macro')

        vaccs.append(val_ac)
        taccs.append(t_ac)

        #if val_ac < 0.5:
        #    print("Skipping gamma {} because of low validation accuracy".format(gamma))
        #    continue

        #print("Dec tree: {}, {} = {:.2f},\t {:.2f},\t {:.2f}".format(res, split, t_ac, val_ac, f1))
        
        clf = svm.SVC(gamma=0.001)
        clf.fit(x_train, y_train)
        st_ac = clf.score(x_train, y_train)
        sval_ac = clf.score(x_val, y_val)
        SVMvaccs.append(val_ac)
        SVMtaccs.append(t_ac)
        predicted = clf.predict(x_test)
        sf1 = metrics.f1_score(y_pred=predicted,y_true=y_test, average='macro')

        print("{} = {:.2f},\t {:.2f},\t {:.2f},\t {:.2f},\t {:.2f},\t {:.2f}".format(split, t_ac, val_ac, f1, st_ac, sval_ac, sf1))
        
       
        cand = {
            'val' : val_ac,
            'train' : t_ac,
            'split' : split,
            'res' : res
        }
        
        op_folder = curr + '/models/decisionTree/dec_{}_{}.joblib'.format(res, split)
        dump(dec, op_folder)
        candidates.append(cand)
      
        
    #ax = fig.add_subplot(4,2,i)
    #plot(ax, vaccs, taccs, split, res)
    i+=1
        
#plt.show()
#best_cand = get_best(candidates, curr)


"""
