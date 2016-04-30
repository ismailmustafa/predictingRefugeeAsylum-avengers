from sklearn import cluster, datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


data = pd.read_csv('data/raw/asylum_clean_full_test.csv')
x = data['sports_score'].values
for i in x:
	print i
y = data['weather_score'].values
print max(x)
print min(x)
plt.scatter(x, y)
plt.show()
# X = np.transpose(np.array([data['sports_score'].values, data['weather_score'].values]))
# print X.shape

# k_means = cluster.KMeans(n_clusters=3)
# k_means.fit(X)
# y = k_means.labels_[::]
# h = .02
# # Create color maps
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, m_max]x[y_min, y_max].
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
# Z = y

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
