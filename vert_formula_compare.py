import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

"""
Creates a 3d scatter plot showing the effect of Cos-Sim and WMD on VERT.

"""

cos_sim = [
    0.6679,0.6692,0.6619,0.638,0.6692,
    0.6692,0.6559,0.66052,0.63047,0.67862,
    0.63514,0.67954,0.68833,0.69517,0.69198,
    0.5065,0.67694,0.66141,0.52076,0.67721,
    0.6164,0.69124,0.6932,0.75007,0.55507,
    0.63431,0.6813,0.69007,0.68639,0.68696,
    0.69974,0.69721,0.70214,0.7079,0.72924,
    0.73254,0.73485,0.7392,0.73857,0.73876,
    0.73523, #.85, 0.9
]

wmd = [
    3.151,3.142,3.146,3.182,3.142,
    3.142,3.0511,3.05722,3.14212,3.01759,
    3.1148,3.10988,3.10448,3.032,3.02534,
    3.58453,3.08507,3.10644,3.54102,3.08009,
    3.18123,3.06001,3.05404,2.72931,3.53581,
    3.12682,2.91779,2.89216,2.85348,2.83245,
    2.82352,2.8238,2.80565,2.77903,2.73715,
    2.73543,2.73376,2.73302,2.73437,2.73907,
    2.76307, #2.2, 1.3
]

formula_1 = []
formula_2 = []
formula_3 = []

for (sim,dis) in zip(cos_sim, wmd):
    print 'Sim:', sim, 'Dis:', dis

    f_1 = float(np.tanh((5.0 *float(sim)) / dis))
    f_2 = float(np.tanh(float(sim) / dis**(1.3.)))
    f_3 = float(.5*(1+ (sim - (dis))) )

    formula_1.append(f_1)
    formula_2.append(f_2)
    formula_3.append(f_3)
    print 'VERT Formula 1:', f_1
    print 'VERT Formula 2:', f_2
    print 'VERT Formula 3:', f_3

print ''
print 'Range f1:', min(formula_1), max(formula_1), max(formula_1)-min(formula_1)
print 'Range f2:', min(formula_2), max(formula_2), max(formula_2)-min(formula_2)
print 'Range f3:', min(formula_3), max(formula_3), max(formula_3)-min(formula_3)


fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(cos_sim, wmd, formula_1, c=formula_1, cmap='Greens');
ax.scatter3D(cos_sim, wmd, formula_2, c=formula_2, cmap='Reds');
ax.scatter3D(cos_sim, wmd, formula_3, c=formula_3, cmap='Blues');

ax.set_xlabel('cos_sim')
ax.set_ylabel('wmd')
ax.set_zlabel('vert f1')
plt.show()
