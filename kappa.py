# -*- coding: UTF-8 -*-
from __future__ import print_function
import codecs
import numpy as np
from sklearn.metrics import cohen_kappa_score
import logging
from sklearn import metrics
import pandas as pd
from scipy.stats import fisher_exact 
from scipy.stats import spearmanr
#fname='../input/raad/HR_tmp.txt'
fname='../input/raad/Hafez_Houman_Raad_Jan2019_no07.txt'
Hlabels=[]
Rlabels=[]
delimiter=','
for line in codecs.open(fname, 'r', 'UTF-8'):
    row = line.split(delimiter)
    Hlabels.extend(row[1])
    Rlabels.extend(row[2])

'''
for line in codecs.open(fname, 'r', 'UTF-8'):
    row = line.split(delimiter)
    Rlabels.extend(row[2])'''
'''
for i in range(len(Rlabels)):
    print(str((Hlabels[i])))
    print(str((Rlabels[i])))'''
with codecs.open( '../input/raad/text_results.txt', 'w', 'UTF-8') as fo:
        #for Y in X_Y_dic:
        fo.write(str(Rlabels) + '\n')

kappa1 = cohen_kappa_score(Hlabels, Rlabels,weights='linear')
kappa2 = cohen_kappa_score(Hlabels, Rlabels)

corr=spearmanr(Hlabels,Rlabels) # hoping teh wardings go away when I have more data
print(str(corr))
print(kappa1,kappa2)
#oddsratio, pvalue = fisher_exact(pd.crosstab(Hlabels, Rlabels))
#print(str(oddsratio))
'''
Rlabels = np.array(Rlabels)
Rlabels = np.maximum(Rlabels, 0).flatten()
Hlabels = np.array(Hlabels)
mad = metrics.mean_absolute_error(Hlabels, Rlabels)
mse = metrics.mean_squared_error(Hlabels, Rlabels)
mape = metrics.mean_absolute_percentage_error(Hlabels, Rlabels)

if verbose:
        print("Mean absolute deviation (MAD) =", mad) 
        print ("Mean squared error (MSE) =", mse)
        print ("Mean absolute percentage error (MAPE) =", mape)
        print ("Cohen kappa score =", kappa1 )
        '''