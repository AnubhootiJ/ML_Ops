import os
from joblib import load

def get_best(candidates, curr):
    best_cand = max(candidates, key=lambda x: x['val'])  
    gamma = best_cand['gamma']     
    op = curr+'/models/model_{}.joblib'.format(gamma)
    mod = load(op)

    print("\nBest model is with Gamma Value = {}, split = {}, and resolution = {} with a validation accuracy of {:.3f}".format(
        best_cand['gamma'], 
        best_cand['split'],
        best_cand['res'],
        best_cand['val']))
    return best_cand