import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

svm_df = pd.read_csv('D:\IITJ\Semester-3\MLOps_HandsON\ML_Ops\svm_output.csv')
gammas = [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05]
train_split = [10,20,30,40,50,60,70,80,90,100]
s_cols = ['Train Data (%)', 'gamma', 'SVMTestAcc', 'SVMValAcc', 'SVMF1Score']

for i, gamma in enumerate(gammas):
    df = svm_df[i*10:i*10+10] 
    #print(df['Train Data (%)'])
    plt.plot(df['Train Data (%)'], df['SVMF1Score'], label = str(gamma))

plt.title('SVM Data Vs. F1 Score')
plt.xlabel('Train Data (%)')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig('results/SVM_Output.jpg')
plt.show()


dt_df = pd.read_csv('D:\IITJ\Semester-3\MLOps_HandsON\ML_Ops\dt_output.csv')
d_cols = ['Train Data (%)', 'MaxDepth', 'DecTestAcc',  'DecValAcc', 'DecF1Score']
depths = [6,8,10,12,14,16]
train_split = [10,20,30,40,50,60,70,80,90,100]

for i, depth in enumerate(depths):
    df = dt_df[i*10:i*10+10] 
    #print(df['Train Data (%)'])
    plt.plot(df['Train Data (%)'], df['DecF1Score'], label = str(depth))

plt.title('Decision Tree Data Vs. F1 Score')
plt.xlabel('Train Data (%)')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig('results/DT_Output.jpg')
plt.show()

