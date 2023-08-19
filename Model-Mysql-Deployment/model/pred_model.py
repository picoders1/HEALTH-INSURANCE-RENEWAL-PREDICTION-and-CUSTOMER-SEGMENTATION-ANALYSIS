####################Prediction ###########
#Part-1

import pandas as pd
#loading dataset Data-version3.csv from google drive
url = 'https://drive.google.com/file/d/1n__aCGFj89JcI3OFaN1Wyt7Hh0XD9iFk/view?usp=sharing'
url2='https://drive.google.com/uc?id=' + url.split('/')[-2]
final_df = pd.read_csv(url2)

#cleaning memory
del(url,url2)

'''dropping [Claim_Cancellation] since final_df after EDA consist only 1 unique value
Reason:15% of dataset should have Claim_Cancellation, 
ratio generated was very less so unique values generated after EDA,
which can be useless for Model prediction '''
#final_df.drop(['Claim_Cancellation'] , axis = 1 , inplace = True)

####################scaling########################
#Part-2

from sklearn.preprocessing import scale
df_scale = scale(final_df)

#####################################PCA##########
# #Part-3

# from sklearn.decomposition import PCA

# #generating PCA for all 34 input features
# pca = PCA(n_components=(34))
# pca_val = pca.fit_transform(df_scale[:,:-1])   #fitting

# #from the screeplot we got a insight that 0-28 features are needed to get 95% of the data
# #selecting only first 29 PCA's
# pca_val = pd.DataFrame(pca_val)
# name=[]
# for i in range(1,35):
#     a="PCA"+str(i)
#     name.append(a)
# pca_val.columns =name 
# pca_final = pd.concat([pca_val.iloc[:, 0:29],final_df.Renewal], axis = 1)

# #cleaning memory
# del(a,df_scale,final_df,i,name,pca,pca_val)
########################Splitting the data #############
#Part-4

#splitting the data into train and test
X = final_df.iloc[:,:-1] # Predictors 
Y = final_df.iloc[:,-1] # Target

#######################KNN############################
#Part-6

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, Y)

########################Pickle#########################
#loading model into pickel
import pickle
pickle.dump(knn,open('pckl_model.pkl','wb'))
######################################END###############