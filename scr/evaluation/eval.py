import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('map')

import pandas as pd
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np

def p_k(true, pred):
    return len(list(set(true) & set(pred))) / len(pred)

#the cup definition => sum_{1}^{min{mapk, n}}{p_k}
#the true definition => frak{1}{num_rel_k} * {sum_{1}^{min{mapk, n}}{p_k * is_doc_k_relevant}}
def ap_k(true, pred, k=3):
    if len(list(set(true) & set(pred[:k]))) < 1:
        return 0
    sum = 0
    for i in range(k):
        sum += p_k(true, pred[:i+1]) * (1 if pred[i] in true else 0)
    return sum / len(list(set(true) & set(pred[:k])))

def mean_ap_k(true, pred, k=3):
    res = 0.
    for (t, p) in zip(true, pred):
        res += ap_k(t, p, k)
    return res/len(true)

def suc_k(true, pred, k=3):
    return int(len(list(set(true) & set(pred[:k]))) > 0)

def mean_suc_k(true, pred, k=3):
    res = 0.
    for (t, p) in zip(true, pred):
        res += suc_k(t, p, k)
    return res/len(true)


