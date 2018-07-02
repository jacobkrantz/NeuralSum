from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

matplotlib.rcParams.update({'font.size': 16})
"""
Plot the relationship between VERT and ROUGE metrics:
    - ROUGE-1
    - ROUGE-2
    - ROUGE-l
Compute the coefficient of determination between them.
23 experiment samples are used. Each of the point values are averages
from runs ranging from 2000 to 2500 testing points.
"""

# ROUGE-1 score
R1 = np.array([
    16.256,16.234,16.014,14.781,16.234,16.234,17.959,18.193,15.781,18.448,
    16.694,17.236,17.690,19.157,19.137,5.6130,17.375,16.578,5.2320,17.585,
    14.266,18.472,18.647
]).reshape(-1, 1)

# ROUGE-2 score
R2 = np.array([
    3.328,3.308,3.282,2.852,3.308,3.308,4.958,5.024,4.311,5.081,
    4.448,4.411,4.416,5.190,5.216,0.283,4.804,4.593,0.330,4.698,
    3.907,4.863,5.049
]).reshape(-1, 1)

# ROUGE-l score (LCS)
Rl = np.array([
    14.624,14.602,14.423,13.344,14.602,14.602,16.269,16.480,14.261,16.667,
    15.179,15.526,15.854,17.140,17.131,5.3110,15.668,14.959,4.9570,15.837,
    13.047,16.546,16.724
]).reshape(-1, 1)

# VERT scores
vert = np.array([
    0.42900,0.43000,0.42600,0.40800,0.43000,0.43000,0.42400,0.42935,
    0.40910,0.44178,0.41219,0.43852,0.44348,0.45104,0.44958,0.31950,
    0.43856,0.42859,0.32854,0.43851,0.39922,0.44741,0.44884,
]).reshape(-1, 1)

def plot(X,Y, xname,yname):
    model = LinearRegression()
    model.fit(X, Y) # x,y
    print("r^2 value:", model.score(X,Y))

    plt.title('VERT-ROUGE Correlation')
    plt.xlabel(xname)
    plt.ylabel(yname)

    plt.scatter(X, Y, color='b', s=50)
    plt.grid()
    plt.plot(X, model.predict(X),color='k') # k r b

    plt.show()

if __name__ == '__main__':
    plot(R1,vert, xname='ROUGE-1', yname='VERT') # R^2 value: 0.954
    plot(R2,vert, xname='ROUGE-2', yname='VERT') # R^2 value: 0.789
    plot(Rl,vert, xname='ROUGE-l', yname='VERT') # R^2 value: 0.947
