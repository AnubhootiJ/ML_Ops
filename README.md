# ML_Ops
=========================================

Quiz - 2, Submitted On: 8/10/2021

1. test_split - Contains two test cases (n=100 and n=9). For each test cases, determining if train:tst:val both for labels and features have required length as per the split. Also, determining if they are quivalent amongst each other (corresponding feature and label). Also ensuring total length match es the number of samples. 22 asserts statements added, 11 for each test case. 

2. test_determine - checks if a model is deterministically giving results by running two different isntances of SVM on same gamma and data and comparing the accuracy and f1-score. 2 asserts statements added.

3. test_model_corrupt - Checks if a stored model is corrupted or not by simply loading the model and checking if it is an SVM object. 1 assert statement added. 


Done with all the 2 test cases as depicted in the original questions,
Added bonus questions as well. The screenshot is attached below -

![plot](results/quiz.png)
