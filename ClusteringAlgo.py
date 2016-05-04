from sklearn import cluster, datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def calculate_kMeans(data, number_of_clusters):
	data_filtered = data[np.isfinite(data['twitter_score'])]
	print len(data_filtered)
	X = np.transpose(np.array([data['sports_score'].values, data['weather_score'].values]))
	k_means = cluster.KMeans(n_clusters=number_of_clusters)
	k_means.fit(X)
	y = k_means.labels_[::]
	return y

# variable up to 8 clusters
def generate_colors(y):
	colors = []
	for i in y:
		if i == 0:
			colors.append('blue')
		elif i == 1:
			colors.append('green')
		elif i == 2:
			colors.append('red')
		elif i == 3:
			colors.append('cyan')
		elif i == 4:
			colors.append('magenta')
		elif i == 5:
			colors.append('yellow')
		elif i == 6:
			colors.append('black')
		else:
			colors.append('white')
	return colors

def plot(data, colors):
	plt.ylim(0.0, 1.0)
	plt.xlim(0.0, 1.0)
	plt.title("mood clustering")
	plt.xlabel("sports score")
	plt.ylabel("weather score")
	plt.scatter(data['sports_score'].values, data['weather_score'].values,color=colors)
	plt.show()

def main():

	# load data
	data = pd.read_csv('data/raw/asylum_clean_full_cluster.csv')
	number_of_clusters = 2 # program can handle 2 - 8
	plot(data, generate_colors(calculate_kMeans(data, number_of_clusters)))

	# # calculate k means clustering for 2-8 clusters
	# mood_kMeans = []
	# for k in np.arange(2,9):
	# 	y = calculate_kMeans(data, k)
	# 	mood_kMeans.append(y)

	# # append columns to asylum data
	# for k,y in enumerate(mood_kMeans):
	# 	key = "mood_k" + str(k+2)
	# 	data[key] = y

	# # write all clustering columns to new data set
	# data.to_csv('data/raw/asylum_clean_full_mood.csv', index=False, index_label=False)


if __name__ == '__main__':
  main()
