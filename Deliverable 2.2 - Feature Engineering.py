##############################  Feature Engineering  ##############################

#Step-1: Reading the data

import pandas as pd

df=pd.read_csv(r'C:\360\Data_Cleaned.csv')
df_new=df.drop(["Claim_Cancellation","Renewal"], axis=1)

#final["Renewal"].value_counts() 

#Step-2: Standardising

from sklearn.preprocessing import scale
df_scale=scale(df_new)

#Step-3: PCA-Principal Component Analysis

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

pca=PCA(n_components=(34)) #Principal component on 35 columns
pca_values=pca.fit_transform(df_scale)
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
print("Variance Ratio: ",var1)

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values
type(pca_values)
pca_data = pd.DataFrame(pca_values)

#Naming the PCA columns
name=[]
for i in range(1,35):
    a="PC"+str(i)
    name.append(a)
pca_data.columns =name 

#Considering only 26 PCA as they it summraise the 91% of the data, so dropping 9 PCA
final = pd.concat([pca_data.iloc[:, 0:25],df.Renewal], axis = 1)
final.to_csv(r'C:\360\Data_feature_engineered.csv', index=False)