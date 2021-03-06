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


class Fista:

    def __init__(self, lambd = 1.0, max_iter = 1000):
        self.lambd = lambd
        self.max_iter = max_iter
        self.coef_ = 0.0
        self.rmse = 0.0
        self.mae = 0.0
        self.current_score = 0.0

    def soft_thresh(self, x, t):
        
        l = len(x)
        y = np.zeros(l)
        for i in range(l):
            if abs(v[i])<=t:
                y[i]=0
            else:
                if v[i]>0:
                    y[i]=v[i]-lam
                elif v[i]<=0:
                    y[i]=v[i]+lam
        
        return y


    def fit(self, A, y):
        
        x = np.zeros(A.shape[1])
        w = x.copy()
        b = 0
        L = linalg.norm(np.dot(A.T, A) ** 2)
        
        for i in range(1000):
            xold = x.copy()
            bold = b
            m = w + (1/L * self.lambd) * np.dot(A.T, (y - np.dot(A, w)))
            x = self.soft_threth(m, self.lambd / L)
            b = 1/2 * (1 + np.sqrt(1 + 4 * (b * b)))
            w = x + ((bold - 1) / (b)) * (x - xold)
            
        self.coef_ = x
        return self

    def predict(self, X):
        y = np.dot(X, self.coef_)
        return y

    
    def score(self, X_test, y_test):
        rmse = np.sqrt(mean_squared_error(y_test, self.predict(X_test)))
        self.rmse = rmse
        mae = mean_absolute_error(y_test, self.predict(X_test))
        self.mae = mae

        self.current_score = 1.253 - (rmse / mae)

        return self.current_score
#        self.scores = np.append(self.scores, self.current_score)

#        return np.mean(self.scores)



if __name__ == '__main__':
    
# init    
    
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
            model = Fista(l, 1000)
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
    model = Fista(param_list[min_index],1000)
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,2:7], data.iloc[:,1], test_size=0.33, shuffle=True, random_state=42)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    print(model.coef_)


