
import pandas as pd # working with data
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
import seaborn as sns  # visualization
import seaborn as sb # visualization
from mpl_toolkits.mplot3d import Axes3D # 3d plot
from termcolor import colored as cl # text customization
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.cluster import KMeans 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


data=pd.read_csv("C:/Users/Dhruv/Desktop/Final_Dataset.csv")

data=data.drop(["Claim_Cancellation"], axis=1)

X = data.values
X = np.nan_to_num(data)
sc = StandardScaler()
 
cluster_data = sc.fit_transform(X)
print(cl('Cluster data samples : ', attrs = ['bold']), cluster_data[:5])

# KMeans________________________________________________________________________________________________
###### scree plot or elbow curve ############
TWSS = []
k = list(range(2,9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(X)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data['cluster_Kmean'] = mb # creating a  new column and assigning it to new column 

clust= data.iloc[:,:].groupby(data.cluster_Kmean).median()


#Plotting Clusters with their segments
markers = ['+', '^','.']
sns.lmplot(x="Age", y="income", data= data, hue = "cluster_Kmean", fit_reg=False, markers = markers,height = 6)

# KMEANS++_____________________________________________________________________________

clusters = 3
model = KMeans(init = 'k-means++', 
               n_clusters = clusters, 
               n_init = 12)
model.fit(X)

labels = model.labels_
print(cl(labels[:100], attrs = ['bold']))

# MODEL INSIGHTS

data['cluster_num'] = labels
print(cl(data.head(), attrs = ['bold']))

print(cl(data.groupby('cluster_num').median(), attrs = ['bold']))

# 3D 
fig = plt.figure(1)
plt.clf()
ax = Axes3D(fig, 
            rect = [0, 0, .95, 1], 
            elev = 48, 
            azim = 134)

plt.cla()
ax.scatter(data['Renewal'], data['Age'], data['income'], 
           c = data['cluster_num'], 
           s = 200, 
           cmap = 'spring', 
           alpha = 0.5, 
           edgecolor = 'darkgrey')
ax.set_xlabel('Renewal', 
              fontsize = 16)
ax.set_ylabel('Age', 
              fontsize = 16)
ax.set_zlabel('Income', 
              fontsize = 16)

plt.savefig('3d_plot.png')
plt.show()

markers = ['+', '^','.']
sns.lmplot(x="Age", y="income", data= data, hue = "cluster_num", fit_reg=False, markers = markers,
         height = 6)


# DBSCAN___________________________________________________________________________

# finding nearest points distance for every row in data
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

# Plotting K-distance Graph
distances = np.sort(distances, axis=0)
distances = distances[:,1]

plt.plot(distances)
plt.xlabel('Data Points sorted by distance', fontsize=14)
plt.ylabel('Epsilon', fontsize=14)
plt.show()

model = DBSCAN(eps=6, min_samples=3)

y = model.fit_predict(X)
# Visualizing all the clusters 
plt.figure(figsize=(25,15))
sns.scatterplot(x=data.Age, y=data.income, 
                hue=y, palette=sns.color_palette('hls', len(np.unique(y))), s=100)
plt.title('Cluster of Customers'.format(data.Age, data.income), size=15, pad=10)
plt.xlabel('data.Age', size=12)
plt.ylabel('data.income', size=12)
plt.legend(loc=0, bbox_to_anchor=[1,1])
plt.show()


# Hierarchical Clustering____________________________________________________________________________

z = linkage(data, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(25,15));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


h_cluster = AgglomerativeClustering(4)
h_cluster.fit(X)
data["h_clusterid"] = h_cluster.labels_
#Plotting Clusters with their segments
markers = ['+', '^','.', '*']
sns.lmplot(x="Age", y="income", data= data, hue = "h_clusterid", fit_reg=False, markers = markers,
         height = 6)



#_______________________________________________________

