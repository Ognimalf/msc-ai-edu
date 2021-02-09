import numpy as np
import pandas as pd
import csv
from sklearn import linear_model
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('mlm.csv')
x_train, y_train, z_train = df.x, df.y, df.z
c = x_train * y_train
X_train = np.column_stack((x_train, y_train))
print(X_train)
z_test = linear_model.ElasticNet()
z_test.fit(X_train, z_train)
print("拟合结果为：Z=%f*X+%f*Y+%f" % (z_test.coef_[0], z_test.coef_[1], z_test.intercept_))

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_train, y_train, z_train)
ax.view_init(elev=20, azim=55)
X, Y = np.arange(0, 100, 1), np.arange(0, 100, 1)
X, Y = np.meshgrid(X, Y)
Z = z_test.coef_[0] * X + z_test.coef_[1] * Y + z_test.intercept_
ax.plot_surface(X, Y, Z, color='r', alpha=0.5)
plt.show()
