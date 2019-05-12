
import numpy as np
import pandas as pd
from scipy import linalg
from math import sqrt
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from spmimage.linear_model import LassoADMM

if __name__ == '__main__':
    data = pd.DataFrame(pd.read_csv('~/Downloads/data.csv'), dtype='float')
    scores = np.zeros(0)
    param_list = [0.001, 0.01, 0.1, 1, 10, 100]
    scoresbyparam = np.zeros(0)
    kf = KFold(n_splits=2)
    for l in param_list:
        print("current lambd is " + str(l))
        for train, test in kf.split(data):
            train_x = data.iloc[train, 2:7]
            train_y = data.iloc[train, 1]
            test_x = data.iloc[test, 2:7]
            test_y = data.iloc[test, 1] 
            model = LassoADMM(l)
            model.fit(train_x, train_y)
            pre = model.predict(test_x)
            rmse = np.sqrt(mean_squared_error(test_y, pre))
            mae = mean_absolute_error(test_y, pre)
            score = 1.253 - (rmse / mae)
            print("score : " + str(score))
            print("coef_ : " + str(model.coef_))
            scores = np.append(scores, score)
        scoresbyparam = np.append(scoresbyparam, np.mean(scores))
    min = np.min(scoresbyparam)
    min_index = np.argmin(scoresbyparam)
    print("best lambd is " + str(param_list[min_index]))
    model = LassoADMM(param_list[min_index])
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,2:7], data.iloc[:,1], test_size=0.33, shuffle=True, random_state=42)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    print(model.coef_)
    
