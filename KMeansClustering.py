import pandas as pd
from sklearn.cluster import KMeans
#import sklearn.cluster.hierarchical as hclust
from sklearn import preprocessing
from matplotlib import pyplot as plt


df = pd.read_csv(r"File to read data from", encoding='latin1', sep = ',')


features = df.drop(['ASIN', 'Product Title'], 1)

data_scaled = preprocessing.normalize(features)
data_scaled = pd.DataFrame(data_scaled, columns = features.columns)

inertia = []

K = range(1,10)
for k in K:
    kmeanmodel = KMeans(n_clusters=k).fit(data_scaled)
    kmeanmodel.fit(data_scaled)
    inertia.append(kmeanmodel.inertia_)

plt.plot(K, inertia, 'bx-')
plt.xlabel('K')
plt.ylabel('Inertia')

plt.show()


kmeans = KMeans(n_clusters=4).fit(data_scaled)

clustered = pd.DataFrame(kmeans.labels_)
clustered.columns= ['Cluster']
clustered['ASIN'] = df['ASIN'].values
clustered['Product Title'] = df['Product Title'].values


ClusteredProducts = pd.concat((features,clustered), 1, join='inner')

ClusteredProducts


ClusteredProducts.to_csv('Name of results file', encoding='utf-8', sep=',')

del df
del features
del data_scaled
del ClusteredProducts
del clustered